import pickle
from modules.ffnn import FFNN

def save_model(model, filepath):
    state = {
        'config': model.config,
        'initializer': model.initializer,
        'weights': [(layer.W.copy(), layer.b.copy()) for layer in model.layers],
        'history': model.history,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    model = FFNN(state['config'], state['initializer'])
    for layer, (W, b) in zip(model.layers, state['weights']):
        layer.W = W
        layer.b = b
    model.history = state['history']
    return model
