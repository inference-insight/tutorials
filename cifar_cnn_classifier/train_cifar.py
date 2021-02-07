import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from cnn_classifier import CNNClassifier

if __name__ == "__main__":

	batch_size = 6
	eval_iter = 5000
	print_iter = 100

	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# CIFAR train dataset
	train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
	                                          shuffle=True, num_workers=2)

	# CIFAR test dataset
	test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
	                                         shuffle=False, num_workers=2)

	classes = train_set.classes

	def imshow(img):
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# get some random training images
	dataiter = iter(train_loader)
	images, labels = dataiter.next()

	# show images
	#imshow(torchvision.utils.make_grid(images))

	model = CNNClassifier(num_classes=len(classes))
	criteria = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for epoch in range(50):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(train_loader):
			inputs, labels = data

			# zero the gradients
			optimizer.zero_grad()

			# forward, backward then update parameters
			outputs = model(inputs)
			prob = F.softmax(outputs)
			loss = criteria(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % print_iter == print_iter-1:
				print('[%d, %5d] loss: %.3f' % \
					(epoch + 1, i + 1, running_loss / print_iter))
				running_loss = 0.0

			# run eval and save checkpoint
			if i % eval_iter == eval_iter-1:    
				print("Running Evaluation...")
				torch.save(model.state_dict(), 'cifar_classifier.pth')

				# do a quick test
				model.eval()
				correct = 0
				total = 0
				for i, data in enumerate(test_loader):
					images, labels = data
					pred = model(images)
					outputs = torch.argmax(pred, axis=-1).detach().numpy()
					probs = F.softmax(pred).detach().numpy()
					correct += np.sum(np.array(np.equal(outputs, labels), dtype=np.int32))
					total += batch_size
					if i == 0:
						print("------------------------Example Test Batch------------------------")
						print('Ground Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
						print('Prediction: ', ' '.join('%5s' % classes[outputs[j]] for j in range(batch_size)))
						print('Confidence: ', ' '.join('%5s' % probs[j][outputs[j]] for j in range(batch_size)))
						print("------------------------------------------------------------------")
						#imshow(torchvision.utils.make_grid(images))
				print("Accuracy:", correct/float(total))
				model.train()
				

