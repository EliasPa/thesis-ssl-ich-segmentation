"""
Original repository: https://github.com/Britefury/cutmix-semisup-seg/


Copyright (c) 2020 University of East Anglia, Norwich, UK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""
import numpy as np

def get_confusion_matrix(targets, predictions, n_classes):
    """
    Calculates confusion matrix.
    Inspired by: https://github.com/Britefury/cutmix-semisup-seg/blob/master/evaluation.py
    """
    targets_ = targets.view(-1).cpu().numpy()
    predictions_ = predictions.view(-1).cpu().numpy()
    return np.bincount(targets_*n_classes + predictions_, minlength=n_classes*n_classes).reshape((n_classes, n_classes))
