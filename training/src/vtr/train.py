import prime_rl._compat  # noqa: F401

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import timedelta

# ruff: noqa: I001

import torch
import torch.distributed as dist
from torch import Tensor

from prime_rl import monitors
from prime_rl.trainer.ckpt import Progress, setup_ckpt_manager
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.sft.data import load_sft_dataset
from prime_rl.trainer.utils import GarbageCollection, clip_grad_norm_, setup_torch_distributed
from prime_rl.trainer.vlm import setup_processor
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.utils.pathing import get_step_path, resolve_latest_ckpt_step
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit

from vtr.config import VTRConfig
from vtr.data import setup_dataloader
from vtr.model import setup_vtr_model
from vtr.objective import degradation, dual_update

REFERENCES = ("full", "blind")
GATE_METRICS = ("rate", "kept_fraction", "gate_mean", "gate_std", "gate_below_0.05", "gate_above_0.95", "gate_entropy")


@dataclass
class VTRProgress(Progress):
    dual: float = 0.0


def gate_metrics(rate: Tensor, kept_fraction: Tensor, probs: Tensor) -> dict[str, Tensor]:
    entropy = -(probs * (probs + 1e-6).log() + (1 - probs) * (1 - probs + 1e-6).log())
    return {
        "rate": rate.detach(),
        "kept_fraction": kept_fraction,
        "gate_mean": probs.mean(),
        "gate_std": probs.std(),
        "gate_below_0.05": (probs < 0.05).float().mean(),
        "gate_above_0.95": (probs > 0.95).float().mean(),
        "gate_entropy": entropy.mean(),
    }


@clean_exit
def train(config: VTRConfig):
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting VTR trainer in {world} (output_dir={config.output_dir})")
    asyncio.run(
        monitors.setup(
            wandb=config.monitors.wandb,
            prime=config.monitors.prime,
            file=config.monitors.file,
            output_dir=config.output_dir,
            producer="trainer",
            run_config=config,
            overview_flavor="sft",
        )
    )
    setup_torch_distributed(timeout=timedelta(seconds=config.dist_timeout_seconds))
    torch.set_float32_matmul_precision(config.matmul_precision)
    torch.manual_seed(config.seed + world.rank)
    resolve_ep(config.model)
    parallel_dims = get_parallel_dims(config.model)
    dp_group = parallel_dims.get_mesh("dp_cp").get_group()
    dp_size = parallel_dims.get_mesh("dp_cp").size()

    assert config.data.batch_size % world.world_size == 0, "data.batch_size must be divisible by the world size"
    grad_accum_steps = config.data.batch_size // world.world_size

    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt, resume=config.resume)
    checkpoint_step = None
    if config.resume is not None:
        checkpoint_step = config.resume.step or resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)

    # The sparse model trains its LM on the routed sequence and its router on R + lambda * D. The full and blind
    # references are independently trained copies of the same initialization that only measure the loss with
    # every and with no routed visual token.
    logger.info(f"Initializing models from {config.model.name}")
    model = setup_vtr_model(config.model, parallel_dims, router=True)
    references = {name: setup_vtr_model(config.model, parallel_dims, router=False) for name in REFERENCES}
    tokenizer = setup_tokenizer(config.tokenizer)
    processor = setup_processor(config.model)

    router_params = list(model.model.router.parameters())
    lm_named_params = [(name, param) for name, param in model.named_parameters() if ".router." not in name]
    lm_params = [param for _, param in lm_named_params if param.requires_grad]
    optimizer, _ = setup_optimizer(config.optim, lm_named_params, parallel_dims, offload="none")
    router_optimizer = torch.optim.AdamW(router_params, lr=config.router.lr, weight_decay=config.router.weight_decay)
    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    reference_optimizers, reference_schedulers = {}, {}
    for name, reference in references.items():
        reference_optimizers[name], _ = setup_optimizer(
            config.optim, list(reference.named_parameters()), parallel_dims, offload="none"
        )
        reference_schedulers[name] = setup_scheduler(
            reference_optimizers[name], config.scheduler, config.max_steps, config.optim.lr
        )

    dataloader = setup_dataloader(config.data, tokenizer, config.renderer, processor)
    val_raw_dataset = load_sft_dataset(config.val.data) if config.val is not None else None

    progress = VTRProgress(dual=config.objective.lambda_init)
    if checkpoint_step is not None:
        logger.info(f"Resuming from checkpoint step {checkpoint_step}")
        skip = ckpt_manager.config
        ckpt_manager.load(
            checkpoint_step,
            model,
            [optimizer, router_optimizer],
            None if skip.skip_scheduler else scheduler,
            progress,
            dataloader=None if skip.skip_dataloader else dataloader,
        )
        for name, reference in references.items():
            ckpt_manager.load_from_path(
                get_step_path(ckpt_manager.ckpt_dir, checkpoint_step) / f"reference_{name}",
                reference,
                [reference_optimizers[name]],
                None if skip.skip_scheduler else reference_schedulers[name],
                None,
            )
        progress.step += 1
    dataiter = iter(dataloader)

    def set_gradient_sync(sync: bool) -> None:
        for module in (model, *references.values()):
            module.set_requires_gradient_sync(sync)

    def forward_backward(micro_batch: dict, *, backward: bool) -> dict[str, Tensor]:
        """Run the four forwards of one sample (full, blind, sparse LM loss, sparse router objective), each followed
        by its backward when training, and return detached per-sample metrics."""
        loss_mask = micro_batch["loss_mask"][0].cuda(non_blocking=True)
        loss_positions = loss_mask.nonzero().squeeze(1)
        mm_kwargs = {key: value.cuda(non_blocking=True) for key, value in micro_batch["mm_kwargs"].items()}
        inputs = dict(
            input_ids=micro_batch["input_ids"].cuda(non_blocking=True),
            mm_token_type_ids=micro_batch["mm_token_type_ids"].cuda(non_blocking=True),
            pixel_values=mm_kwargs["pixel_values"],
            image_grid_thw=mm_kwargs["image_grid_thw"],
            loss_positions=loss_positions,
            loss_targets=micro_batch["target_ids"][0].cuda(non_blocking=True)[loss_positions],
            block_size=config.router.block_size,
            ce_chunk_size=config.objective.ce_chunk_size,
        )

        full = references["full"](**inputs, routing="full")
        inputs["image_embeds"] = full.image_embeds.detach()
        if backward:
            (full.ce.mean() / grad_accum_steps).backward()
        blind = references["blind"](**inputs, routing="blind")
        if backward:
            (blind.ce.mean() / grad_accum_steps).backward()

        # Two independent gate samples: the first trains the LM (and the router through the keep probabilities
        # that scale the kept features), the second gives the router objective. The LM is frozen during the
        # second so that R + lambda * D updates only the router.
        sparse = model(**inputs, routing="sparse")
        if backward:
            (sparse.ce.mean() / grad_accum_steps).backward()
            for param in lm_params:
                param.requires_grad_(False)
        routed = model(**inputs, routing="sparse")
        l_full, l_blind, l_sparse = full.ce.detach(), blind.ce.detach(), routed.ce
        D = degradation(l_full, l_blind, l_sparse)
        router_loss = routed.rate + progress.dual * D
        if backward:
            (router_loss / grad_accum_steps).backward()
            for param in lm_params:
                param.requires_grad_(True)

        l_sparse = l_sparse.detach()
        zero = torch.zeros((), device="cuda")
        metrics = {
            "loss": sparse.ce.detach().mean(),
            "l_full": l_full.mean(),
            "l_blind": l_blind.mean(),
            "l_sparse": l_sparse.mean(),
            "D": D.detach(),
            "router_loss": router_loss.detach(),
            "degradation_sum": (l_sparse - l_full).sum(),
            "image_gain_sum": (l_blind - l_full).sum(),
            "num_tokens": torch.tensor(float(loss_positions.numel()), device="cuda"),
            "routed": torch.tensor(float(routed.keep_probs is not None), device="cuda"),
        }
        if routed.keep_probs is not None:
            metrics.update(gate_metrics(routed.rate, routed.kept_fraction, routed.keep_probs))
        else:
            metrics.update({name: zero for name in GATE_METRICS})
        return metrics

    def all_reduce_sum(values: dict[str, Tensor]) -> dict[str, float]:
        stacked = torch.stack(list(values.values()))
        dist.all_reduce(stacked, group=dp_group)
        return dict(zip(values, stacked.tolist()))

    @torch.no_grad()
    def validate(step: int) -> None:
        loader = setup_dataloader(
            config.val.data, tokenizer, config.renderer, processor, max_epochs=1, raw_dataset=val_raw_dataset
        )
        batches = iter(loader)
        keys = ("loss", "l_full", "l_blind", "l_sparse", "degradation_sum", "image_gain_sum", "kept_fraction")
        totals = {key: torch.zeros((), device="cuda") for key in (*keys, "num_tokens", "routed")}
        while True:
            # Ranks can hold different numbers of rows; every rank stops once any rank runs out.
            micro_batch = next(batches, None)
            has_data = torch.tensor(int(micro_batch is not None), device="cuda")
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0:
                break
            metrics = forward_backward(micro_batch, backward=False)
            tokens, routed = metrics["num_tokens"], metrics["routed"]
            for key in ("loss", "l_full", "l_blind", "l_sparse"):
                totals[key] += metrics[key] * tokens
            totals["degradation_sum"] += metrics["degradation_sum"]
            totals["image_gain_sum"] += metrics["image_gain_sum"]
            totals["kept_fraction"] += metrics["kept_fraction"] * routed
            totals["num_tokens"] += tokens
            totals["routed"] += routed
        totals = all_reduce_sum(totals)
        tokens = max(totals["num_tokens"], 1.0)
        val_metrics = {
            "val/loss": totals["loss"] / tokens,
            "val/l_full": totals["l_full"] / tokens,
            "val/l_blind": totals["l_blind"] / tokens,
            "val/l_sparse": totals["l_sparse"] / tokens,
            "val/degradation_ratio": totals["degradation_sum"] / (totals["image_gain_sum"] + 1e-6),
            "val/kept_fraction": totals["kept_fraction"] / max(totals["routed"], 1.0),
            "step": step,
        }
        logger.success(
            f"Validation | Step {step} | Loss {val_metrics['val/loss']:.4f} | "
            f"r {val_metrics['val/degradation_ratio']:+.4f} | kept {100 * val_metrics['val/kept_fraction']:.1f}%"
        )
        asyncio.run(monitors.log(val_metrics, step=step))

    def save_checkpoint(step: int) -> None:
        for name, reference in references.items():
            ckpt_manager.save_to_path(
                get_step_path(ckpt_manager.ckpt_dir, step) / f"reference_{name}",
                reference,
                [reference_optimizers[name]],
                reference_schedulers[name],
                progress,
            )
        ckpt_manager.save(step, model, [optimizer, router_optimizer], scheduler, progress, dataloader=dataloader)
        ckpt_manager.maybe_clean()

    max_norm = math.inf if config.optim.max_norm is None else config.optim.max_norm
    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None
    logger.info(f"Starting training loop (max_steps={config.max_steps}, grad_accum_steps={grad_accum_steps})")
    while progress.step <= config.max_steps:
        if gc_handler is not None:
            gc_handler.run(progress.step)
        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()

        sums = None
        for micro_step in range(grad_accum_steps):
            # Gradients are reduced across ranks only in the last micro step's backwards.
            set_gradient_sync(micro_step == grad_accum_steps - 1)
            metrics = forward_backward(next(dataiter), backward=True)
            sums = metrics if sums is None else {key: sums[key] + value for key, value in metrics.items()}
        set_gradient_sync(True)
        sums = all_reduce_sum(sums)

        num_samples = grad_accum_steps * dp_size
        routed = max(sums["routed"], 1.0)
        step_metrics = {f"train/{key}": sums[key] / num_samples for key in sums if key not in GATE_METRICS}
        step_metrics |= {f"train/{key}": sums[key] / routed for key in GATE_METRICS}
        degradation_ratio = sums["degradation_sum"] / (sums["image_gain_sum"] + 1e-6)
        step_metrics["train/degradation_ratio"] = degradation_ratio
        step_metrics["train/dual"] = progress.dual
        progress.dual = dual_update(progress.dual, degradation_ratio, config.objective)

        router_grad_norm = torch.nn.utils.get_total_norm([param.grad for param in router_params]).full_tensor()
        # The router is clipped jointly with the sparse LM, then stepped by its own optimizer.
        grad_norm = clip_grad_norm_(None, model, max_norm)
        optimizer.step()
        optimizer.zero_grad()
        router_optimizer.step()
        router_optimizer.zero_grad()
        reference_grad_norms = {}
        for name, reference in references.items():
            reference_grad_norms[name] = clip_grad_norm_(None, reference, max_norm)
            reference_optimizers[name].step()
            reference_optimizers[name].zero_grad()
            reference_schedulers[name].step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0:
            if progress.step < config.max_steps:
                logger.info(f"Saving checkpoint at step {progress.step}")
                save_checkpoint(progress.step)
        if config.val is not None and progress.step % config.val.interval == 0:
            validate(progress.step)

        step_time = time.perf_counter() - step_start
        step_metrics |= {
            "optim/lr": lr,
            "optim/grad_norm": grad_norm.item(),
            "optim/router_grad_norm": router_grad_norm.item(),
            **{f"optim/{name}_grad_norm": value.item() for name, value in reference_grad_norms.items()},
            "perf/peak_memory": torch.cuda.max_memory_reserved() / 1024**3,
            "time/step": step_time,
            "step": progress.step,
        }
        logger.success(
            f"Step {progress.step} | {format_time(step_time)} | Loss {step_metrics['train/loss']:.4f} | "
            f"Grad. Norm {step_metrics['optim/grad_norm']:.4f} | kept {100 * step_metrics['train/kept_fraction']:.1f}% "
            f"| L_full {step_metrics['train/l_full']:.3f} L_blind {step_metrics['train/l_blind']:.3f} "
            f"L_sparse {step_metrics['train/l_sparse']:.3f} | R {step_metrics['train/rate']:.3f} "
            f"D {step_metrics['train/D']:+.3f} r {degradation_ratio:+.3f} | lambda {step_metrics['train/dual']:.3f}"
        )
        asyncio.run(monitors.log(step_metrics, step=progress.step))
        if progress.step == config.max_steps:
            break
        progress.step += 1

    if config.ckpt is not None:
        logger.info(f"Saving final checkpoint at step {progress.step}")
        save_checkpoint(progress.step)
    logger.success("VTR trainer finished")
    asyncio.run(monitors.finalize())


def main():
    set_proc_title("VTRTrainer")
    train(cli(VTRConfig))


if __name__ == "__main__":
    main()
