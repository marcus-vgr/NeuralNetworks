import numpy as np
import pickle
import copy


####################################### LAYERS ############################################

''' Input Layer Class - To be used in the Model class. It is just a Layer that get the input data and do 
a forward pass to the first hidden Layer, without any activation function,
'''
class Layer_Input:

    def forward(self, inputs, training):
        self.output = inputs

''' Dense Layer Class - The hidden layers in our Neural Network. Here we also pass values for L1 and L2 regularizations
in the case it is useful. Remember: regularizations are used to (kind of) avoid some neurons to just memorize the 
training data.
'''
class Layer_Dense:

    #Initializaton
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
    bias_regularizer_l1=0, bias_regularizer_l2=0):
        #Initialize weights and biases randomly
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))  # Initialize biases with a shape
        # Set regulators
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        #Remember Inputs
        self.inputs = inputs
        # Calculate output
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #Gradient on Parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradient on regularization
        # L1 regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        #L2 regularization
        if self.weight_regularizer_l2 > 0:
            self.dweights +=  2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l2 > 0:
            self.dbiases +=  2 * self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


''' Dropout Layer: To be used during the training process. We randomly drop some neurons in each iteration in 
the training to avoid memorization.
'''
class Layer_Dropout:

    def __init__(self, rate):
    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
        self.rate = 1-rate

    def forward(self,inputs,training):
        #Remember inputs
        self.inputs = inputs

        #If not in training mode, do not drop any neuron
        if not training:
            self.output = inputs.copy()  # Copy to use a different location in the memory.
            return

        #Generate and save scaled mask with 0 or 1
        self.binary_mask = np.random.binomial(1,self.rate, size=inputs.shape) / self.rate
        #Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask



############################################ ACTIVATION FUNCTIONS ##############################################

''' Activation_ReLU: Usually used in hidden layers
'''

class Activation_ReLU:
    
    def forward(self, inputs, training):
        #Remember Inputs
        self.inputs = inputs
        #Output
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        #Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs. To be used with accuracy
    def predictions(self, outputs):
        return outputs

''' Activation Softmax: Usually used in the final layer to obtain a probability distribution in the predictions
'''

class Activation_Softmax:
    
    def forward(self, inputs, training):
        # Remeber Input
        self.inputs = inputs
        
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # We included the subtraction to avoid
        # large values in the exponential
        
        # Normalize the for each sample 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def backward(self, dvalues): 
        
        #Create unitialized array
        self.dinputs = np.empty_like(dvalues)
        
        #Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1,1)
            #Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflag(single_output) - np.dot(single_output, single_output.T)
            #Calculate simple-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs. To be used with accuracy
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    
''' Activation Sigmoid: Usually used in place of ReLU in Regression 
'''
            
class Activation_Sigmoid:

    def forward(self, inputs, training):
        #Remember input and calculate output
        self.inputs = inputs
        self.outputs = 1/ (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        # (outputs > 0.5) returns a list of True and False, that are converted to 0 or 1 when we multiply by 1
        return (outputs > 0.5) * 1

''' Activation Linear: Do the job of ReLU
'''

class Activation_Linear:

    def forward(self, inputs, training):
        #Remember input and calculate output
        self.inputs = inputs
        self.outputs = inputs
    
    def backward(self,dvalues):
        #Gradient on values
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

############################################## OPTIMIZERS ####################################################

class Optimizer_SGD:
    
    # Initialize optimizer - set settings
    # learning rate of 1 is default
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    # Call once before any parameter updates    
    def pre_update_params(self):
        if self.decay: # if self.decay != returns True, otherwise returns False
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        
    # Update parameters
    def update_params(self, layer):
        
        if self.momentum: #Check if there is a momentum, i.e, momentum != 0
            # If Layer does not contain momentum arrays, createm them filled with zeros
            # The hasattr() method returns true if an object has the given named attribute 
            #and false if it does not.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            # Build biases
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentum = bias_updates
         
        # Vanilla SGD updates (without momentum)
        else:
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases
        
        #Update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
class Optimizer_Adagrad:
    
    # Initialize optimizer - set settings
    # learning rate of 1 is default
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    # Call once before any parameter updates    
    def pre_update_params(self):
        if self.decay: # if self.decay != returns True, otherwise returns False
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        
    # Update parameters
    def update_params(self, layer):
        
        # If Layer does not contain momentum arrays, createm them filled with zeros
        # The hasattr() method returns true if an object has the given named attribute 
        #and false if it does not.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
            
            
        # Vanilla SGD updates + normalization
        layer.weights += -self.current_learning_rate * layer.dweights \
        / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases \
        / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
        
class Optimizer_RMSprop:
    
    # Initialize optimizer - set settings
    # learning rate of 1 is default
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    # Call once before any parameter updates    
    def pre_update_params(self):
        if self.decay: # if self.decay != returns True, otherwise returns False
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        
    # Update parameters
    def update_params(self, layer):
        
        # If Layer does not contain momentum arrays, createm them filled with zeros
        # The hasattr() method returns true if an object has the given named attribute 
        #and false if it does not.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2
            
            
        # Vanilla SGD updates + normalization
        layer.weights += -self.current_learning_rate * layer.dweights \
        / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases \
        / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
        
class Optimizer_Adam:
    
    # Initialize optimizer - set settings
    # learning rate of 1 is default
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    # Call once before any parameter updates    
    def pre_update_params(self):
        if self.decay: # if self.decay != returns True, otherwise returns False
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)
        
    # Update parameters
    def update_params(self, layer):
        
        # If Layer does not contain momentum arrays, createm them filled with zeros
        # The hasattr() method returns true if an object has the given named attribute 
        #and false if it does not.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
                
        # Update momentums with squared current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases 
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))
        
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations+1))
            
        # Vanilla SGD updates + normalization
        layer.weights += -self.current_learning_rate * weight_momentums_corrected \
        / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected \
        / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
    
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


########################################### LOSS ###############################################################

''' Commom Loss Class. We separate loss into regularization and data Loss. We also calculate loss with 
trainable layers.
'''
class Loss:

    #Regularization Loss calculation
    def regularization_loss(self):
        
        regularization_loss = 0
        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            #Regularization weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)

            #Regulatization bias
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)

        return regularization_loss
    
    # Set/Remember trainable_layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers  = trainable_layers
        
     # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output,y, *,include_regularization=False):
        
        #Calculate sample  losses
        sample_losses = self.forward(output,y)
        
        #Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count -- to be used with batches
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):

        #Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    #Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0



''' CrossEntropy Loss: Used with categorical data
'''

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        #Number of samples in a batch
        samples = len(y_pred)
        
        #Clip data to avoid Logs with small arguments
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Probabilities for target values  - only if categotical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods        
    
    def backward(self, dvalues, y_true):
        
        #Number of Samples
        samples = len(dvalues)
        #Number of labels in every sample
        #We will use the first sample to count them
        labels = len(dvalues[0])
        
        #If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #Calculate Gradient
        self.dinputs = - y_true / dvalues
        #Normalize gradient
        self.dinputs = self.dinputs / samples

''' Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
'''
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs/samples        

''' Binary CrossEntropy Loss: Usually used in Regression along with Sigmoid
'''

class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):

        #As in Categortical Cross Entropy we clip the data to avoid overflow
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        #Calculate sample-wise loss
        sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample
        outputs = len(dvalues[0])
        #Clip data
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        #Gradient over values
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples

''' Mean Squared Error Loss
'''

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2 , axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample
        outputs = len(dvalues[0])

        #Gradient over values
        self.dinputs = - 2 * (y_true - dvalues) / outputs
        #Normalize
        self.dinputs = self.dinputs / samples

''' Mean Absolute Error Loss
'''

class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean(np.abs(y_true - y_pred) , axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)
        #Number of outputs in every sample
        outputs = len(dvalues[0])

        #Gradient over values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        #Normalize
        self.dinputs = self.dinputs / samples


############################################ ACCURACY #######################################################

''' Commom accuracy class (like in Loss)
'''
class Accuracy:

    #Calculates an accuracy given prediciton and true values
    def calculate(self, predictions, y):

        #Get comparison results
        comparisons = self.compare(predictions, y)
        #Calculate mean accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of losses and sample count -- to be used with batches
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):

        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_count = 0
        self.accumulated_sum = 0

''' Accuracy calculation for classification model
'''
class Accuracy_Categorical(Accuracy):

    #No initialization is needed
    def init(self,y):
        pass

    #Compare
    def compare(self, predictions,y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

''' Accuracy calculation for regression model
'''
class Accuracy_Regression(Accuracy):

    def __init__(self):
        #Create precision property 
        self.precision = None

    # Calculate precision value based on passed ground truth values
    '''we’ll calculate the standard deviation from the ground truth target values and then divide it by ​ 250 . ​ This
    value can certainly vary depending on your goals. The larger the number you divide by, the more “strict” the accuracy 
    metric will be. ​ 250 ​ is our value of choice. '''

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
        
    def compare(self,predictions, y):
        return np.absolute(predictions - y) < self.precision


########################################### MODEL CLASS #############################################################

class Model:

    def __init__(self, save_stats=False):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
        self.save_stats = save_stats
        if self.save_stats:
            self.lr_arr = []
            self.loss_arr = []
            self.acc_arr = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # Set Loss, optimizer and Accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        #Create and set the input layer
        self.input_layer = Layer_Input()
        
        #Count all the objects 
        layer_count = len(self.layers)
        
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        
        #Iterate all objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Update Loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        '''If output activation is Softmax and  loss function is Categorical Cross-Entropy create an object 
        of combined activation and loss function containing faster gradient calculation'''
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            # Create object combining activation and loss for faster backpropagation
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    #Train the model
    def train(self, X, y, *,epochs=1, batch_size = None, 
    print_every_epoch=1, print_every_step=0, validation_data = None):

        # Initialize accuracy object
        self.accuracy.init(y)

        if validation_data is not None:
            X_val, y_val = validation_data

        # Default value if batch size is not being set
        train_steps = 1
        validation_steps = 1
        #Calculate number of steps when we have a batch_size
        if batch_size is not None:
            train_steps = len(X) // batch_size  #This is the integer division. For instance, 5 // 2 = 2.
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        #Training loop
        for epoch in range(1,epochs+1):

            #print(f'epoch: {epoch}')

            #Reset accumulated loss and accuracy
            self.loss.new_pass()
            self.accuracy.new_pass()

            #Iterate over steps
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                #forward pass
                output = self.forward(batch_X, training=True)
                #calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                #Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                #Backward pass
                self.backward(output, batch_y)

                #Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                #Print a summary
                if print_every_step != 0:
                    if not step % print_every_step or step == train_steps - 1:
                        print(f'step: {step}, '+
                        f'acc: {accuracy:.3f}, '+
                        f'loss: {loss:.3f} ('+
                        f'data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), '+
                        f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            if self.save_stats:
                self.lr_arr.append(self.optimizer.current_learning_rate)
                self.acc_arr.append(epoch_accuracy)
                self.loss_arr.append(epoch_loss)


            if print_every_epoch != 0:
                if not epoch % print_every_epoch or epoch == 1 or epoch == epochs:
                    print(f'Epoch {epoch} training, '+
                    f'acc: {epoch_accuracy:.3f}, '+
                    f'loss: {epoch_loss:.3f} ('+
                    f'data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), '+
                    f'lr: {self.optimizer.current_learning_rate}')


                    #If there is a validade data
                    if validation_data is not None:
                        #Reset accumulated loss and accuracy
                        self.loss.new_pass()
                        self.accuracy.new_pass()

                        #Iterate over steps
                        for step in range(validation_steps):
                            if batch_size is None:
                                batch_X = X_val
                                batch_y = y_val
                            else:
                                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                                batch_y = y_val[step*batch_size:(step+1)*batch_size]

                            #forward pass
                            output = self.forward(batch_X, training=True)
                            #calculate loss
                            self.loss.calculate(output, batch_y, include_regularization=True)
                            loss = data_loss + regularization_loss
                            #Get predictions and calculate accuracy
                            predictions = self.output_layer_activation.predictions(output)
                            accuracy = self.accuracy.calculate(predictions, batch_y)
                        
                        validation_loss = self.loss.calculate_accumulated()
                        validation_accuracy = self.accuracy.calculate_accumulated()
                        print(f'VALIDATION, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

        
    def forward(self,X,training):

        self.input_layer.forward(X,training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            
            self.softmax_classifier_output.backward(output,y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return
        
        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    
    # Saves the model
    def save(self, path):

        #Make a deep copy of current model instance
        model = copy.deepcopy(self)

        #Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        #Remove data from input layer and gradient from loss object
        model.input_layer.__dict__.pop('output',None)
        model.loss.__dict__.pop('dinputs',None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
            
        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Load a Model
    @staticmethod    
    def load(path):
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model