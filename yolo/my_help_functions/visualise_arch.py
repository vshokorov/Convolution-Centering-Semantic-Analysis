import json
import numpy as np
import torch


class ArchVisualiser():
    def __init__(self, file_name):
        with open(file_name, 'r') as fr:
            model_arch = json.load(fr)
            self.convs_ids = model_arch['convs_ids']
            self.connections = model_arch['connections']
        
        self.forward_order = [c[0] for c in self.convs_ids]
    
    def check_forward_order(self, infer_engine):
        x = torch.randn(1, 3, *infer_engine.infer_size)
        forward_order = []

        def get_verbouse_hook(name):
            def hook(module, input, output):
                forward_order.append(name)
            return hook

        hooks = []
        for name, module in infer_engine.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(get_verbouse_hook(name)))

        with torch.no_grad():
            infer_engine.model(x)

        for h in hooks: h.remove()

        assert len(self.convs_ids) == len(forward_order)

    def __call__(self, ax):

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        def get_y(row_id):
            if ax.get_yscale() == 'linear':
                pad = (y_max - y_min) / 15
                return y_min - pad * row_id
            else:
                pad = (np.log(y_max) - np.log(y_min)) / 15
                return np.exp(np.log(y_min) - pad * row_id)

        # Draw points
        for i, ln in enumerate(self.convs_ids):
            color = 'red' if ln[0].startswith('head') else 'black'
            if x_min <= i <= x_max:
                ax.plot([i], [get_y(ln[1])], 'o', color=color, markersize=4)

        for con in self.connections:
            if con[2] <= x_min or con[0] >= x_max:
                continue
            ax.annotate(
                "", xy=(min(x_max, con[2]), get_y(con[3])), xytext=(max(x_min, con[0]), get_y(con[1])),
                arrowprops=dict(arrowstyle="-", lw=1.0, color='darkblue', connectionstyle=con[4])
            )
        
        ax.set_xlabel('Conv Layer Number')
        
        ax.set_xticks(np.arange(len(self.convs_ids)), minor=True)
        ax.grid(which='both')

        ax.set_ylim(get_y(3), y_max)
        ax.set_xlim(x_min, x_max)