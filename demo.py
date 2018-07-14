import torch
from matplotlib import pyplot as plt

def showchannel(mat):
	# mat has shape [n, c, w, h]
	n, c, w, h = mat.shape
	flatten = mat[0].reshape(c, w*h)
	plt.imshow(flatten.t())
	plt.plot((3.5,3.5), (-0.5,3.5), "red")
	plt.plot((7.5,7.5), (-0.5,3.5), "red")


if __name__ == "__main__":
	x = torch.ones((10, 12, 2, 2))
	
	x[:,0:4,:,:] *= 1
	x[:,4:8,:,:] *= 2
	x[:,8:12,:,:] *= 3

	y = x.reshape(10, 3, 4, 2, 2)
	# y = y.swapaxes(1, 2) 
	y = torch.transpose(y, 1, 2)
	y = y.reshape(10, 12, 2, 2)

	plt.subplot(211)
	plt.title("before")
	showchannel(x)

	plt.subplot(212)
	plt.title("after")
	showchannel(y)

	plt.show()