import torch
import torch.nn as nn

from .st_gcn import STGCN
from .graph import Graph
from .facenet import FaceNet

class MultiModalModel (nn.Module):
    def __init__(self, numclasses, face_model, skeleton_model):
        super(MultiModalModel, self).__init__()
        
        self.NumClasses = numclasses
        self.FaceModel = FaceNet(inchanels=face_model['inchanels'],
            outchanels=face_model['outchanels'])

        G = Graph(skeleton_model['Glabel'])
        self.SkeletonModel = STGCN(channel=skeleton_model['channel'],
            num_class=skeleton_model['num_classes'],
            window_size=skeleton_model['window_size'],
            num_point=skeleton_model['num_point'],
            use_data_bn=skeleton_model['use_data_bn'],
            graph=G,
            mask_learning=skeleton_model['mask_learning'],
            use_local_bn=skeleton_model['use_local_bn'],
            dropout=skeleton_model['dropout']
            )


    def forward(self, x1, x2):
        # outs = []
        # for i,M in enumerate(self.Modals):
        #     outs.append(M.forward(X[i]))
        x1 = self.FaceModel.forward(x1)
        x2 = self.SkeletonModel.forward(x2)

        x = torch.cross(x1,x2)
        return x , [x1,x2]

# class MultiModalModel (nn.Module):
#     def __init__(self, numclasses, modals=[]):
#         super(MultiModalModel, self).__init__()

#         models = []
#         for mdl in modals:
#             if mdl['model'] == 'FaceNet':
#                 fn = FaceNet(inchanels=mdl['inchanels'], outchanels=mdl['outchanels'])
#                 models.append(fn)
#             elif mdl['model'] == 'st-gcn':
#                 G = Graph(mdl['Glabel'])
#                 gcn = STGCN(channel=mdl['channel'], 
#                     num_classes=mdl['num_classes'],
#                     window_size=mdl['window_size'], 
#                     num_point=mdl['num_point'],
#                     graph=G,
#                     dropout=mdl['dropout'],
#                     mask_learning=mdl['mask_learning'],
#                     use_data_bn=mdl['use_data_bn']
#                     )
#                 models.append(gcn)
#             else:
#                 raise ValueError()
#         self.Modals = models        

#     def forward(self, x1, x2):
#         # outs = []
#         # for i,M in enumerate(self.Modals):
#         #     outs.append(M.forward(X[i]))
#         x1 = self.Modals[0].forward(x1)
#         x2 = self.Modals[1].forward(x2)

#         x = torch.cross(x1,x2)
#         return x , [x1,x2]
