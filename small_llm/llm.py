import json
from dataclasses import dataclass
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import trackio
import pickle


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree
Array = chex.Array


@chex.dataclass(frozen=True)
class SampleConfig:
    max_new_tokens: int
    temperature: float
    top_k: int


@dataclass
class ModelConfig:
    d_model: int
    d_mlp: int
    n_layers: int
    n_heads: int
    max_len: int
    dropout: float
    dtype: jnp.dtype
    seed: int
    learning_rate: float
    weight_decay: float
    run_name: str
    sample_config: SampleConfig
    vocab_size: int = None

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config = json.load(f)
        if "sample_config" in config:
            sample_config = SampleConfig(**config["sample_config"])
            config["sample_config"] = sample_config
        return cls(**config)


@chex.dataclass(frozen=True)
class ModelState:
    params: Params
    opt_state: optax.OptState
    step: int


class MLP(nn.Module):
    d_model: int
    d_mlp: int
    dropout: float

    @nn.compact
    def __call__(self, x: Array, training: bool) -> Array:
        x = nn.Dense(features=self.d_mlp)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(features=self.d_model)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        return x


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x: Array, mask: jnp.ndarray, training: bool) -> Array:
        x = nn.LayerNorm()(x)
        x = x + nn.SelfAttention(qkv_features=self.d_model,
                                 num_heads=self.n_heads)(x, mask=mask)
        x = nn.LayerNorm()(x)
        x = x + MLP(d_model=self.d_model, d_mlp=self.d_model,
                    dropout=self.dropout)(x, training)
        return x


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int

    @nn.compact
    def __call__(self) -> Array:
        pos = jnp.arange(self.max_len)[:, None]
        i = jnp.arange(self.d_model)[None, :]
        angle = pos / jnp.power(10000, 2 * (i // 2) / self.d_model)
        pe = jnp.where(i % 2 == 0, jnp.sin(angle), jnp.cos(angle))
        return pe[None, :, :]


class GPT(nn.Module):
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    max_len: int
    dropout: float

    def setup(self):
        self.token_emb = nn.Embed(
            num_embeddings=self.vocab_size, features=self.d_model)
        self.pos_emb = nn.Embed(
            num_embeddings=self.max_len, features=self.d_model)

    @nn.compact
    def __call__(self, tokens: Array, training: bool = True) -> Array:
        bsz, seq_len = tokens.shape
        pos = jnp.arange(seq_len)[None]
        x = self.token_emb(tokens)   # (B, seq_len, d_model)
        x = x + self.pos_emb(pos)  # (B, seq_len, d_model)
        x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        attn_mask = nn.make_causal_mask(tokens, dtype=jnp.bool_)
        for i in range(self.n_layers):
            x = TransformerBlock(d_model=self.d_model, n_heads=self.n_heads, dropout=self.dropout)(
                x, attn_mask, training)
        x = nn.Dense(features=self.vocab_size)(x)  # (B, seq_len, vocab_size)
        return x


class LLM():

    def __init__(self, config: ModelConfig):
        self.config = config
        self.key = jax.random.PRNGKey(self.config.seed)

        self.model = GPT(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            max_len=self.config.max_len,
            dropout=self.config.dropout,
        )
        rng_init = self._next_rng()
        _, params = self.model.init_with_output(rng_init, jnp.ones(
            (1, self.config.max_len), jnp.int16))

        # optimizer
        self.optimizer = optax.adamw(
            learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # model state
        self.model_state = ModelState(
            params=params,
            opt_state=self.optimizer.init(params),
            step=0,
        )

        def loss_fn(params: Params, rng: PRNGKey, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            logits = self.model.apply(
                params,
                x,
                training=True,
                rngs={"dropout": rng},
            )

            # Mask loss to ignore padding in targets
            per_tok_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, y)  # (B, T)
            target_nonpad = (y != 0).astype(per_tok_loss.dtype)
            loss = (per_tok_loss * target_nonpad).sum() / \
                jnp.maximum(target_nonpad.sum(), 1.0)
            return loss

        self.loss_fn = jax.value_and_grad(loss_fn)

        # train step
        def _train_step(state: ModelState,
                        batch: tuple[jnp.ndarray, jnp.ndarray],
                        rng: PRNGKey) -> ModelState:
            params = state.params
            step = state.step
            x, y = batch[:, :-1], batch[:, 1:]
            loss, grads = self.loss_fn(params, rng, x, y)
            updates, opt_state = self.optimizer.update(
                grads, state.opt_state, params)
            params = optax.apply_updates(params, updates)
            state = state.replace(
                params=params, opt_state=opt_state, step=step + 1)
            return state, loss

        self.train_step = jax.jit(_train_step)

        # eval step
        def _eval_step(state: ModelState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            params = state.params
            x, y = batch[:, :-1], batch[:, 1:]
            logits = self.model.apply(params, x, training=False)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, y).mean()
            return loss

        self.eval_step = jax.jit(_eval_step)

    def train(self, train_data: jnp.ndarray, val_data: jnp.ndarray, num_steps: int):
        trackio.init(project="fake-training", name=self.config.run_name, config={
            "epochs": num_steps,
        })
        for step in range(num_steps):
            batch = next(train_data)
            self.model_state, loss = self.train_step(self.model_state, batch, rng=self._next_rng())
            val_loss = self.eval_step(self.model_state, next(val_data))
            trackio.log({
                "epoch": step,
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
            })
            trackio.finish()

    def generate(self, prompt: Array):
        max_new_tokens = self.config.sample_config.max_new_tokens
        b_size, prompt_len = prompt.shape
        buffer = jnp.zeros((b_size, max_new_tokens), dtype=jnp.int16)
        buffer = jnp.concatenate([prompt, buffer], axis=1)

        # TODO: the model should be applied in the context window, not on the
        # whole buffer.
        for i in range(max_new_tokens - self.config.max_len -1 ):
            ctx = buffer[:, i:i + self.config.max_len]
            logits = self.model.apply(
                self.model_state.params, ctx, training=False)  # (B, T, V)
            # (B, V)  -- we only predict the last (next) token.
            logits = logits[:, -1, :] / self.config.sample_config.temperature
            v, _ = jax.lax.top_k(
                logits, self.config.sample_config.top_k)  # (B, K)
            logits = jnp.where(
                logits < v[:, -1], jnp.finfo(self.config.dtype).min, logits)
            next_token = jax.random.categorical(self._next_rng(), logits)  # (B,)
            decoding_step = i + prompt_len
            buffer = buffer.at[:, decoding_step].set(next_token)  # (B, T)
        return buffer

    def _next_rng(self):
        self.key, rng = jax.random.split(self.key)
        return rng

    def save(self, path: str):
        pickle.dump(self.model_state.params, open(path, "wb"))

    def load(self, path: str):
        params = pickle.load(open(path, "rb"))
        self.model_state = self.model_state.replace(params=params)
