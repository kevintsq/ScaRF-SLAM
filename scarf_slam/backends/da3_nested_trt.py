"""DA3NESTED-GIANT-LARGE-1.1 (ScaRF's default depth model) with its two networks as TensorRT engines and the nested combination in torch.
Drop-in for `DepthAnything3.model` (the api keeps its input/output processing): called as
    model(image [1,N,3,H,W], extrinsics_norm [1,N,4,4], intrinsics [1,N,3,3], export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy)
and returns the same addict Dict (depth, depth_conf [1,N,H,W], extrinsics [1,N,3,4], intrinsics [1,N,3,3], is_metric, scale_factor).
Engines: DA3NESTED_anyview_<N>x<H>x<W>.onnx and DA3NESTED_metric_<N>x<H>x<W>.onnx, exported from the checkpoint outside this repo and
built / cached per GPU by TrtRunner on first use. The engines are static in N: the batch size `views` is loaded at start, and a batch with
another number of views (e.g. a caller that adds older keyframes as context) uses the engines exported for that N, loaded on first use. The post-network steps are the model's own methods (they use no weights):
metric branch sky quantile step, metric scaling by the predicted intrinsics, least-squares depth alignment, sky handling."""
import os, sys, time
import torch
from addict import Dict
from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
from scarf_slam.backends.trt_runner import TrtRunner


class DA3NestedTrt(torch.nn.Module):
    def __init__(self, onnx_dir: str, views: int = 6, height: int = 378, width: int = 504):
        super().__init__()
        self.onnx_dir, self.height, self.width = onnx_dir, height, width; self.engines = {}
        self.t_any = self.t_met = self.t_post = 0.0; self.n = 0
        self.register_buffer("_device_anchor", torch.zeros(1, device="cuda"))   # DepthAnything3._get_model_device() looks for a parameter/buffer
        self._engines_for(views)

    def _engines_for(self, views):
        """(anyview, metric) TrtRunners for a batch of `views` images, loaded (built and cached on first use) when first needed"""
        if views not in self.engines:
            tag = f"{views}x{self.height}x{self.width}"; paths = [os.path.join(self.onnx_dir, f"DA3NESTED_{net}_{tag}.onnx") for net in ("anyview", "metric")]
            if not all(os.path.exists(p) or os.path.exists(p.replace(".onnx", "_fp16.onnx")) for p in paths):   # TrtRunner takes the _fp16 copy on TensorRT >= 11
                raise FileNotFoundError(f"no DA3 nested ONNX for {views} views in {self.onnx_dir} (export it with --views {views})")
            self.engines[views] = tuple(TrtRunner(p, workspace_gb=8) for p in paths)
            print(f"[scarf] DA3 nested TensorRT engines ({views} views): {', '.join(os.path.basename(r.engine_path) for r in self.engines[views])}", flush=True)
        return self.engines[views]

    @torch.inference_mode()
    def forward(self, x, extrinsics=None, intrinsics=None, export_feat_layers=None, infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced"):
        assert x.shape[0] == 1 and tuple(x.shape[2:]) == (3, self.height, self.width), f"engines are (1, N, 3, {self.height}, {self.width}), got {tuple(x.shape)}"
        anyview, metric = self._engines_for(int(x.shape[1]))
        assert extrinsics is not None and intrinsics is not None, "the nested TensorRT path needs camera extrinsics/intrinsics (ScaRF always passes the VIO poses)"
        with torch.autocast("cuda", enabled=False):
            t0 = time.time(); d, c, e, k = anyview(x.float(), extrinsics.float(), intrinsics.float()); torch.cuda.synchronize(); self.t_any += time.time() - t0
            t0 = time.time(); md, ms = metric(x.float()); torch.cuda.synchronize(); self.t_met += time.time() - t0
            t0 = time.time()
            out = Dict(depth=d, depth_conf=c, extrinsics=e, intrinsics=k)
            mout = DepthAnything3Net._process_mono_sky_estimation(None, Dict(depth=md, sky=ms))   # the metric branch's in-model sky step
            out = NestedDepthAnything3Net._apply_metric_scaling(None, out, mout)
            out = NestedDepthAnything3Net._apply_depth_alignment(None, out, mout)
            out = NestedDepthAnything3Net._handle_sky_regions(None, out, mout)
            out.aux = {}
            self.t_post += time.time() - t0; self.n += 1
        return out

    def timing(self):
        n = max(self.n, 1); return f"nested-trt per submap: anyview {1000*self.t_any/n:.0f} ms, metric {1000*self.t_met/n:.0f} ms, post {1000*self.t_post/n:.0f} ms ({self.n} submaps)"
