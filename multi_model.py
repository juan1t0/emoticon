import torch
import torch.nn as nn

from st_gcn import Model as STGCN
from graph import Graph
from facenet import FaceNet

class MultiModalModel (nn.Module):
    def __init__(self, numclasses, modals=[]):
        super(MultiModalModel, self).__init__()

        self.Modals = createModels(modals)

    def createModels(self, modals):
        models = []
        for mdl in modals:
            if mdl['model'] == 'FaceNet':
                fn = FaceNet(inchanels=mdl['inchanels'], outchanels=mdl['outchanels'])
                models.append(fn)
            elif mdl['model'] == 'st-gcn':
                G = Graph
                gcn = STGCN(channel=mdl['channel'], 
                    num_classes=mdl['num_classes'],
                    window_size=mdl['window_size'], 
                    num_point=mdl['num_point'],
                    graph=G
                    )
                models.append(gcn)
            else:
                raise ValueError()

        return models

    def forward(self, x1, x2):
        # outs = []
        # for i,M in enumerate(self.Modals):
        #     outs.append(M.forward(X[i]))
        x1 = self.Modals[0].forward(x1)
        x2 = self.Modals[1].forward(x2)

        x = torch.cross(x1,x2)
        return x , [x1,x2]



'''
channel, num_classes, window_size, num_point, 
num_person=1, use_data_bn=False,
backbone_config=None, graph=None,
graph_args=dict(), mask_learning=False,
use_local_bn=False, multiscale=False,
temporal_kernel_size=9, dropout=0.5):
'''