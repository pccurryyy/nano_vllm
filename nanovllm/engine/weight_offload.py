"""Inference-only decoder weight residency window.

Reference design: HuangRZzzz/nano-vllm-GPU-offload main c25959d (MIT).
Unlike moving whole modules back and forth, immutable CPU master weights are
retained; evicting GPU copies needs no device-to-host transfer. Runtime buffers
and KV pools stay on the execution device. Dynamic parameter storage requires
eager, uncompiled execution in this initial implementation.
"""
import torch


class WeightOffloadManager:
    def __init__(self, layers, num_layers, window_size=1, device="cuda"):
        if not 0 < num_layers <= len(layers) or window_size < 1:
            raise ValueError("Invalid weight offload layer count or window size")
        self.layers = layers
        self.start = len(layers) - num_layers
        self.window = min(window_size, num_layers)
        self.device = torch.device(device)
        self.masters = {}
        self.active = set()
        self.stats = dict(h2d_copies=0, h2d_bytes=0, peak_active_layers=0)
        for i, layer in enumerate(layers):
            if i < self.start:
                layer.to(self.device)
            else:
                masters = []
                for param in layer.parameters():
                    master = param.detach().to("cpu").clone()
                    if self.device.type == "cuda":
                        master = master.pin_memory()
                    param.data = master
                    masters.append((param, master))
                self.masters[i] = masters
                # RoPE caches are runtime buffers, not decoder weight payload.
                for module in layer.modules():
                    for name, buffer in module._buffers.items():
                        if buffer is not None:
                            module._buffers[name] = buffer.to(self.device)
        self.cpu_bytes = sum(master.numel() * master.element_size()
                             for entries in self.masters.values() for _, master in entries)

    def _evict(self, indices):
        for i in indices:
            for param, master in self.masters[i]:
                param.data = master
        self.active.difference_update(indices)

    def begin_forward(self):
        if self.active:
            self.end_forward()

    def prepare_layer(self, i):
        if i < self.start:
            return self.layers[i]
        target = set(range(i, min(len(self.layers), i + self.window)))
        self._evict(self.active - target)
        for j in sorted(target - self.active):
            for param, master in self.masters[j]:
                param.data = master.to(self.device, non_blocking=False)
                self.stats["h2d_copies"] += 1
                self.stats["h2d_bytes"] += master.numel() * master.element_size()
            self.active.add(j)
        self.stats["peak_active_layers"] = max(self.stats["peak_active_layers"], len(self.active))
        return self.layers[i]

    def end_forward(self):
        self._evict(set(self.active))
