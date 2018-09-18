class Person():
    def p(self):
        self.a = 11
        print(self.a)

    def __call__(self, n):
        print("n = ", n)
# p = Person()
# p.p()
# p(34)

import torch

a = [[[1,2,3],[4,15,6]],[[7,8,9],[10,11,12]]]
a = torch.Tensor(a)
print(a)
a = torch.max(a,0)
print(a)