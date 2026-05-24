import numpy as np
import torch
import torchviz
import matplotlib.pyplot as plt
from collections import deque
import re


__all__ = ['ArchVisualiser']


class Node():
    name = None             # Имя, которое соответствует torch module
    tag = None              # Имя в torch autograd, по сути тег операции
    id = None               # id, которое задает каждой ноде torchviz
    row_id = None           # уровень вершины по слоям. Скипы и подобное 
                            # идет выше чем основной стрим
    is_model_block = False  # Флаг, что эта вершина есть является скелетом 
                            # forward прохода модели

    def __init__(self, id):
        self.id = id
        self.forward_nodes = []
    
    def set_tag(self, tag):
        assert self.tag is None
        self.tag = tag
        
        if '\n' in tag:
            name = tag.split("\n", 1)[0]

            if len(name) > 0:
                name = re.match(r'(^.+)(\.bias|\.weight)$', name)
                assert name is not None, repr(tag)
                name = name.group(1)
            
                self.name = name
    
    def __repr__(self):
        return (f'Node(id={self.id}, tag={self.tag}, name={self.name}, '
                f'forward:{len(self.forward_nodes)}, row_id={self.row_id}), '
                f'is_model_block={self.is_model_block}'
        )


connections_type = {
    '00': "arc3,rad=0.",                 # прямое соединение между свертками
    '11': "angle3,angleA=45,angleB=135", # skip connection, через верх
    '01': "angle3,angleA=90,angleB=0",   # подъем наверх для shortcut
    '10': "angle3,angleA=0,angleB=110",  # спуск вниз после shortcut
}





class ArchVisualiser():
    def __init__(self, 
        start_node_name,
        end_node_tag=None,
        filter_names=['ConvolutionBackward0'], 
    ):
        self.start_node_name = start_node_name
        self.end_node_tag = end_node_tag
        self.filter_names = filter_names
        self.n_rows = 0
        self.__start_node_id = None
        self.__end_node_id = None
    
    def __is_to_plot_node(self, node):
        return node.tag in self.filter_names

    def init_from_file(self, file_name):
        with open(file_name, 'r') as fr:
            lines = fr.read()
        self.__init_from_source(lines)
    
    def init_from_model(self, model, input_shape, device):
        x = torch.randn(1, *input_shape, device=device)
        
        model.eval()
        output = model(x)

        self.dot = torchviz.make_dot(output.mean(), params=dict(model.named_parameters()), )
        self.__init_from_source(self.dot.source)

    
    def __init_from_source(self, lines):
        self.__nodes = {}

        # Строку-граф -> список вершин
        for l in lines.split('\t'):
            l = l.rstrip()
            node_id = re.match(r'^\d{15}', l)
            if node_id  is None:
                print('ArchVisualiser: skip:', l)
                continue
            node_id = node_id.group(0)
            if node_id not in self.__nodes:
                self.__nodes[node_id] = Node(id=node_id)

            pair_node_id = re.match(r'^\d{15} -> (\d{15})', l)
            label = re.match(r'^\d{15} \[label=("([^"]*)"|([^ \]]+))', l)
            if pair_node_id is not None:
                pair_node_id = pair_node_id.group(1)

                self.__nodes[node_id].forward_nodes.append(self.__nodes[pair_node_id])
            
            
            elif label is not None:
                assert (label.group(2) is None) or (label.group(3) is None)
                label = label.group(2) or label.group(3)
                
                self.__nodes[node_id].set_tag(label)
            
            else:
                print('ArchVisualiser: ArchVisualiser: unmatched:', l)

        # Проверка, что каждая вершина имеет tag
        for n in self.__nodes.values():
            assert n.tag is not None
        
        # Находим первую и последнюю вершину скелета модели
        for n in self.__nodes.values():
            if n.tag == self.end_node_tag:
                assert self.__end_node_id is None
                self.__end_node_id = n.id
            elif n.name == self.start_node_name:
                assert self.__start_node_id is None
                self.__start_node_id = n.id
        assert self.__start_node_id is not None

        # Проходим поиском в ширину для определения скелета модели
        dfs_visited = set()
        dfs_queue = deque([self.__start_node_id])
        while dfs_queue:
            node_id = dfs_queue.popleft()
            node = self.__nodes[node_id]
            node.is_model_block = True
            dfs_visited.add(node_id)

            for n_node in node.forward_nodes:
                if n_node.id not in dfs_visited:
                    dfs_queue.append(n_node.id)
        if self.__end_node_id is None:
            print(f'ArchVisualiser: No end_node_id find!, Set end_node: {node}')
            self.__end_node_id = node_id
        del dfs_visited
        del dfs_queue

        node = self.__nodes[self.__start_node_id]
        while (node.id != self.__end_node_id):

            if self.__is_to_plot_node(node):
                self.__start_node_id = node.id
                print(f'ArchVisualiser: Move start_node_id forward, to: {node}')
                break
            else:
                node.is_model_block = False

            assert len(node.forward_nodes) == 1, node
            node = node.forward_nodes[0]
        else:
            raise BaseException('No blocks found for plot')
            

            

        # Прокидываем имена вперед. Изначально имена имеют только веса модели, 
        # их нужно прокинуть вперед до скелета модели
        def move_name_forward(node_id):
            node = self.__nodes[node_id]

            if (node.name is None) or node.is_model_block:
                return

            assert len(node.forward_nodes) == 1, node
            for next_node in node.forward_nodes:
                assert (next_node.name is None) or (next_node.name == node.name)
                next_node.name = node.name
                move_name_forward(next_node.id)

        for n in self.__nodes:
            move_name_forward(n)
        

        # Заполняем массивы, который будут использоваться для отрисовки
        self.convs = []
        self.connections = []
        # Проход по графу вперед, пока не дойдем до развилки или конца
        def go_forward(node, row_id):
            n_steps = 0
            while (node.id != self.__end_node_id) and (node.tag != 'AddBackward0'):
                
                assert node.row_id is None
                node.row_id = row_id
                self.n_rows = max(node.row_id, self.n_rows)
                if self.__is_to_plot_node(node):
                    assert node.is_model_block
                    assert node.id not in self.convs
                    self.convs.append(node)
                    n_steps += 1

                while len(node.forward_nodes) > 1:
                    node, skip_steps = process_fork(node, row_id)
                    n_steps += skip_steps
                
                if len(node.forward_nodes) == 0:
                    print('ArchVisualiser: model ends at', node)
                    break
                node = node.forward_nodes[0]

            if n_steps > 0:
                self.connections.append(
                    (len(self.convs)-n_steps, row_id, len(self.convs)-1, row_id, '00')
                )
            return node, n_steps

        def process_fork(node, row_id):
            assert len(node.forward_nodes) == 2
            fork_start = len(self.convs)

            long_node, short_node = node.forward_nodes

            if long_node.tag == 'AddBackward0':
                long_node, short_node = short_node, long_node
            elif 'shortcut' in long_node.name:
                long_node, short_node = short_node, long_node
            
            short_node, s_steps = go_forward(short_node, row_id + 1)
            long_node,  l_steps = go_forward(long_node, row_id)
            assert short_node == long_node
            assert s_steps <= l_steps
            assert len(self.convs) == fork_start + s_steps + l_steps
            
            if s_steps == 0:
                self.connections.append((fork_start-0.5, row_id, len(self.convs)-0.5, row_id, '11'))
            else:
                self.connections.append((fork_start-0.5, row_id, fork_start, row_id+1, '01'))
                self.connections.append((fork_start+s_steps-1, row_id+1, len(self.convs)-0.5, row_id, '10'))
            
            self.connections.append((fork_start-1, row_id, fork_start+s_steps, row_id, '00'))

            return short_node, s_steps + l_steps
        
        node, n_steps = go_forward(self.__nodes[self.__start_node_id], 0)
        if self.__end_node_id is not None:
            assert node.id == self.__end_node_id
        assert n_steps == len(self.convs)

    
    @property
    def forward_names(self):
        return [c.name for c in self.convs]

    def __call__(self, ax):

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        def get_y(row_id):
            l = self.n_rows - row_id
            if ax.get_yscale() == 'linear':
                pad = (y_max - y_min) / 5 / self.n_rows
                return y_min - pad * l
            else:
                pad = (np.log(y_max) - np.log(y_min)) / 5 / self.n_rows
                return np.exp(np.log(y_min) - pad * l)

        # Draw points
        for i, n in enumerate(self.convs):
            color = 'red' if n.name.startswith('head') else 'black'
            if x_min <= i <= x_max:
                ax.plot([i], [get_y(n.row_id)], 'o', color=color, markersize=4)

        for con in self.connections:
            if con[2] <= x_min or con[0] >= x_max:
                continue
            ax.annotate(
                "", xy=(min(x_max, con[2]), get_y(con[3])), xytext=(max(x_min, con[0]), get_y(con[1])),
                arrowprops=dict(arrowstyle="-", lw=1.0, color='darkblue', connectionstyle=connections_type[con[4]])
            )
        
        ax.set_xlabel('Conv Layer Number')
        
        ax.set_xticks(np.arange(len(self.convs)), minor=True)
        ax.grid(which='both')

        ax.set_ylim(get_y(-0.5), y_max)
        ax.set_xlim(x_min, x_max)

if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(20, 8))

    import sys
    sys.path.append('/home/samosyuk/vsh_phd_tests/')
    from pytorch_cifar100.models import resnet
    model = resnet.resnet50()
    archvis = ArchVisualiser('conv1.0', 'MeanBackward1')
    archvis.init_from_model(model, (3, 32, 32), 'cpu')

    ax.plot(np.random.random(len(archvis.convs)))
    archvis(ax)
    ax.legend()

    plt.savefig('tmp.png')

    archvis.dot.format = 'svg'
    archvis.dot.render('model_arch')


    # import sys
    # sys.path.append('/home/samosyuk/DDA/arcface_torch')

    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"


    # from backbones import get_model

    # backbone = get_model(
    #     'r50', dropout=0.0, fp16=True, num_features=512).cuda(0)


    # archvis = ArchVisualiser('conv1', 'MeanBackward1')
    # archvis.init_from_model(backbone, (3, 112, 112), 'cuda')

    # ax = plt.subplot()
    # ax.plot(np.random.random(len(archvis.convs)))
    # archvis(ax)
    # ax.legend()

    # plt.savefig('tmp.png')
