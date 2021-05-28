import unittest
import torch
import numpy as np
import cle.cle as cle

def manual_CLE(f, f_hat, labels, N, n_classes, tau, background_index=0):

    batch_wide_sum = 0
    for i in range(f.shape[0]):
        labels_ = labels[i].view(-1)
        f_ = f[i].permute(1,2,0).view(-1, n_classes)
        f_hat_ = f_hat[i].permute(1,2,0).view(-1, n_classes)
        not_bg = ~torch.eq(labels_, background_index)
        labels_ = labels_[not_bg]
        f_ = f_[not_bg]
        f_hat_ = f_hat_[not_bg]

        dots = torch.zeros((f_.shape[0],f_.shape[0]))
        for x in range(f_.shape[0]):
            for y in range(f_.shape[0]):
                for d in range(n_classes):
                    dots[x,y] += (f_[x,d] * f_hat_[y,d]) / tau

        outer_sum = 0 
        for i in range(f_.shape[0]):
            inner_sum = 0
            N_yi = 0
            y_i = labels_[i]
            for j in range(f_.shape[0]):
                y_j = labels_[j]
                if y_i == y_j:
                    N_yi += 1
                    top = torch.exp(dots[i,j])
                    bottom = torch.tensor([torch.exp(dots[i,k]).item() for k in range(f_.shape[0])]).sum()
                    inner_sum += torch.log(top / bottom)
            outer_sum += inner_sum / (N_yi + 0.01)
            
        outer_sum = - outer_sum / N
        batch_wide_sum += outer_sum

    return batch_wide_sum / f_hat.shape[0]

class CLETest(unittest.TestCase):

    def test_output_is_correct_case_1(self):
        # setup
        n_classes=2
        w = 2
        h = 2
        b_size = 1
        N = w*h
        labels = torch.ones((b_size,w,h))#(torch.rand(b_size,w,h)*n_classes).long()
        f = torch.tensor([
            [
                [
                    [1,0.5],
                    [0.8,0],
                ],
                [
                    [1,1.5],
                    [1.1,1.0],
                ]
            ]
        ])#torch.rand((b_size,n_classes,w,h))
        f_hat = torch.tensor([
            [
                [
                    [1,0.5],
                    [0.8,0],
                ],
                [
                    [1,1.5],
                    [1.1,1.0],
                ]
            ]
        ])#torch.rand((b_size,n_classes,w,h))

        print(labels)
        # act
        tau = 0.5
        loss = cle.CLE(f=f, f_hat=f_hat, labels=labels, N=N, n_classes=n_classes,tau=tau, down_sample_factor=1)['loss']
        
        # assert
        manual_loss = manual_CLE(f=f, f_hat=f_hat, labels=labels, N=N, n_classes=n_classes,tau=tau)
        print("MANUAL", manual_loss)
        np.testing.assert_almost_equal(np.array([manual_loss.item()]), np.array([loss.item()]), decimal=5)


    # Allowed to fail if the square root of the number of classes not in background is not even
    def test_batch_wise_output_is_correct_case_1(self):
        # setup
        n_classes=2
        w = 2
        h = 2
        b_size = 1
        N = w*h
        labels = (torch.rand(b_size,w,h)*n_classes).long()
        f = torch.rand((b_size,n_classes,w,h))
        f_hat = torch.rand((b_size,n_classes,w,h))

        # act
        tau = 0.5
        loss = cle.batch_wise_CLE(f=f, f_hat=f_hat, labels=labels, N=N, n_classes=n_classes,tau=tau, down_sample_factor=1)['loss']
        # assert
        manual_loss = manual_CLE(f=f, f_hat=f_hat, labels=labels, N=N, n_classes=n_classes,tau=tau)
        np.testing.assert_almost_equal(np.array([manual_loss.item()]), np.array([loss.item()]), decimal=5)

if __name__ == '__main__':
    unittest.main()
