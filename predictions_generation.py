from DSOLFUTR.models.model1_resnet_ox import first_head
from DSOLFUTR.models.model2_resnet import second_head

model1_weights_path='model1.ckpt'
model2_weights_path='model2.ckpt'

if __name__ == "__main__":

	model1 = first_head()
	model1.load_weights(model1_weights_path)

	model2 = second_head()
	model2.load_weights(model2_weights_path)