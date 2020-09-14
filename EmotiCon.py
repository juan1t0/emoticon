import torch
import torch.nn as nn

class EmoticonModel(nn.Module):
    def __init__(self, numclass, 
        multmodel = None,
        contmodel = None,
        deepmodel = None):
        
        super(EmoticonModel, self).__init__()
        self.MultiModel = multmodel
        self.ContextModel = contmodel
        self.DepthModel = deepmodel

        self.NClasses = numclass

        self.LinearLayers(nn.Sequential(
            nn.Linear(self.NClasses * 3, self.NClasses * 2),
            nn.Linear(self.NClasses * 2, self.NClasses)))
    
    def forward(self, x):
        x1,mx = self.MultiModel.forward(x['face_landmarks'], x['selketon_pose'])
        x2 = self.ContextModel.forward(x['context'])
        x3 = self.DepthModel.forward(x['depth'])

        main_x = torch.cat((x1, x2, x3))
        main_x = self.LinerarLayers(main_x)
        main_x = main_x.view(main_x.size(0), -1)
        
        return mx, main_x
