import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data, loader_client, NOIIDD, dirichlet_partition
from data import commun_test_set
from tqdm import tqdm


torch.cuda.is_available()
# dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)



# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Initialize the model, loss function, and optimizer

    def fit(self,epochs,train_loader):
# Training the model
        self.to(dev)
        for epoch in tqdm(range(epochs)):
            self.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(dev), y_batch.to(dev)
                self.optimizer.zero_grad()
                outputs = self(X_batch).squeeze(dim=1)  # Predict
                loss = self.criterion(outputs, y_batch)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                
            # if (epoch+1) % 10 == 0:
                # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")



    def evaluation(self,test_loader):
        self.eval()
        correct = 0
        total = 0
        i = 0
        #self.to(dev)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(dev), target.to(dev)
                outputs = self(data).squeeze(dim=1)
                y_pred_class = (outputs >= 0.5).float() 
                total += target.size(0)
                correct += (y_pred_class == target).sum().item()
                i+=1
                if i > 5:# this for the batches
                    break
        return correct/total
    

    def get_model_grads(self):
        return [param.grad for param in self.parameters()]
    
    
    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_parameters(self,params):
         for model_parametro, param in zip(self.parameters(), params):
             model_parametro.data = param


    def save_model(self):
        """Saves the global model for a given round."""
        torch.save(self.state_dict(), "./model_weights.pth")

    def load_model(self):
        """Loads the global model for a given round."""
        self.load_state_dict(torch.load("./model_weights.pth",weights_only=True))
        self.to(dev)


if __name__ == "__main__":
    features=30
    data= dirichlet_partition()
    commoon=commun_test_set(data)
    modelo = LogisticRegressionModel(features)
    modelo.fit(3,data[0][0])
    #modelo.load_model()
    print(modelo.evaluation(data[1][1]))

    # data=load_data()
    # clientes=loader_client(data[0],data[1],[0.1, 0.3, 0.9])
    # # data=  noisy_clients([1.0,3.0,5.0])
    # # clientes= data_loaders(data)

    # # common_dataset=commun_test_set(clientes)
    # model = LogisticRegressionModel(features)
    # model.fit(5,clientes[2][0])
    # print(model.evaluation(clientes[2][1]))
    # #model.reset_parameters()