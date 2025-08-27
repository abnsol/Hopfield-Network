import numpy as np

class HopfieldNetwork:
  def __init__(self,patterns):
    self.patterns = np.array(patterns) #multiple memory / Fixed points
    self.num_neurons = self.patterns.shape[1] if self.patterns.size > 1 else len(self.patterns) # works for single/multiple memory

    # network
    self.state = np.random.randint(-1,2,(self.num_neurons, 1))      # random state for starter
    self.weights = np.zeros((self.num_neurons, self.num_neurons) )  # weight matrix
    self.energies = []                                              # energy

  def train(self):
    self.weights = (1/ self.patterns.shape[0]) * self.patterns.T @ self.patterns
    np.fill_diagonal(self.weights,0)

  def update_rule(self,n_updates):
    for neuron in range(n_updates):
      idx = np.random.randint(0,self.num_neurons)       # select random neuron
      activate = np.dot(self.weights[idx,:],self.state) #activation
      self.state[idx] = 1 if activate >= 0 else -1      #threshold function / UPDATE RULE
  
  def predict(self, x, max_steps=100):
    self.state = x.reshape(-1,1).copy()
    prev = None
    for _ in range(max_steps):
        self.update_rule(self.num_neurons) 
        
        # no change 
        if prev is not None and np.array_equal(self.state, prev):
            break  

        prev = self.state.copy()
    return self.state.flatten()

  def compute_energy(self): #compute energy
        energy = -0.5*np.dot(np.dot(self.state.T,self.weights),self.state)
        self.energies.append(energy)
    