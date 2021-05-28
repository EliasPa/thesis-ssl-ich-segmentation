import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import operator
import torchvision.transforms as transforms
import numpy as np

class MTResNet(nn.Module):

    # TODO: No dropout implemented
    def __init__(self, n_classes=10):
        super(MTResNet, self).__init__()
        self.student = models.resnet18(pretrained=False, num_classes=n_classes)
        self.teacher = models.resnet18(pretrained=False, num_classes=n_classes)
        self.first = True

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30, scale=(0.8, 1.2), translate=(0.1,0.1), shear=5),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor()
        ])

        self.alpha = 0.0

    def forward(self, x):

        device = x.device
        view1 = torch.zeros(x.shape)
        view2 = torch.zeros(x.shape)
        for i, x_i in enumerate(x):
            view1[i, :] = self.transforms(x_i.cpu()) # stochastic
            view2[i, :] = self.transforms(x_i.cpu()) # stochastic

        view1 = view1.to(device)
        view2 = view2.to(device)

        student_output = self.student(view1)

        with torch.no_grad():
            teacher_output = self.teacher(view2)

        return view1, view2, student_output, teacher_output

    def calc_alpha(self, training_step, training_steps):
        x = training_step / training_steps
        sigmoid_fac = np.e**(-5*(1-x)**2)
        start_alpha = 0.99
        target_alpha = 0.999
        delta = target_alpha - start_alpha

        return start_alpha + sigmoid_fac * delta

    def calc_ema(self, previous_mean, incoming_value, alpha=0.99):
        return alpha * incoming_value + (1 - alpha) * previous_mean

    def ema(self, alpha):
        self.alpha = alpha
        for teacher_p, student_p in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_p.data = self.calc_ema(teacher_p.data, student_p.data, alpha=alpha)

    def get_latest_alpha(self):
        return self.alpha

    def optimize(self, forward_output, training_step, training_steps):
        _, _, student_prediction, teacher_prediction = forward_output
        loss = F.mse_loss(student_prediction, teacher_prediction, reduction='sum')
        self.ema(self.calc_alpha(training_step, training_steps)) # this seems to be off by one optimizer update
        return loss
