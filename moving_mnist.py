import socket
import numpy as np
from torchvision import datasets, transforms
from torch import randint

class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.digit_size = image_size//2
        self.deterministic = deterministic
        self.seed_is_set = False 
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
    


class MovingMNIST_unidir_random_axis(object):

    """Data Handler that creates uniderectional MNIST dataset on the fly."""
    '''         Each video contains motion in only one direction         '''

    def __init__(self, train, data_root, seq_len=10, num_digits=1, image_size=64, steps=[-15,-10,10,15]):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 28
        self.seed_is_set = False 
        self.channels = 1
        self.steps = steps

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = self.steps[randint(0, len(self.steps),(1,1)).item()]
            dy = self.steps[randint(0, len(self.steps),(1,1)).item()]
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    dy = -dy
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    dy = -dy
                if sx < 0:
                    sx = 0
                    dx = -dx
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    dx = -dx

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x
    
class MovingMNIST_custom_step(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=1, image_size=64, step=10, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size/2)
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1
        self.step = step

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        step = self.step
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-step+1, step)
            dy = np.random.randint(-step+1, step)
            for t in range(self.seq_len):
                dx = np.random.randint(-step+1, step)
                dy = np.random.randint(-step+1, step)
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, step)
                        dx = np.random.randint(-step+1, step)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-step+1, 0)
                        dx = np.random.randint(-step+1, step)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, step)
                        dy = np.random.randint(-step+1, step)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-step+1, 0)
                        dy = np.random.randint(-step+1, step)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x

class MovingMNIST_random_axis(object):


    def __init__(self, train, data_root, seq_len=3, num_digits=1, image_size=64, step = 5):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size / 2)
        self.seed_is_set = False 
        self.channels = 1
        self.step = step
        self.directions = ['up', 'down', 'left', 'right']

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def set_direction(self, direction):
        if direction == 'up':
            dx = 0
            dy = self.step
        elif direction == 'down':
            dx = 0
            dy = -self.step
        if direction == 'left':
            dx = -self.step
            dy = 0
        elif direction == 'right':
            dx = self.step
            dy = 0
        return dx, dy

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            
            #chose random axis to move
            direction = np.random.choice(self.directions)
            #we start at the center
            sx = digit_size - digit_size//2
            sy = digit_size - digit_size//2
            
            dx, dy = self.set_direction(direction)
            #print(dx,dy)
            
            for t in range(self.seq_len):
                #print(digit.shape)
                #print(x.shape)
                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x

    
class MovingMNIST_unidir_2_axis(object):

    """
    Data Handler that creates uniderectional MNIST dataset on the fly.
    Each video contains motion in only one direction         
    Digits starting centered                     
    Moving in 2 axis 
    """
    def __init__(self, train, data_root, seq_len=3, num_digits=1, image_size=64, step = [11, 5]):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size / 2)
        self.seed_is_set = False 
        self.channels = 1
        self.steps = step
        self.directions = ['up', 'down', 'left', 'right']
        self.data = datasets.MNIST(
                    path,
                    train=train,
                    download=True,
                    transform=transforms.Compose(
                    [transforms.Resize(self.digit_size),
                     transforms.ToTensor()]))
        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def set_initial_position(self, direction):
        sx = sy = (self.image_size - self.digit_size) // 2
        return sx, sy

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            digit_steps = self.steps[:]
            
            #sample direction
            direction = np.random.choice(self.directions)
            
            #select initial location based on direction
            sx, sy = self.set_initial_position(direction)
            
            #place digit at selected initial location
            x[0, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
            
            for t in range(1, self.seq_len):
                np.random.shuffle(digit_steps)
                step = digit_steps.pop()
                if direction == 'up':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)   
                elif direction == 'down':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                elif direction == 'right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 1)
                elif direction == 'left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 1)
            #x[x>1] = 1.
        return x    
    
    
    
        
class MovingMNIST_unidir_4_axis_centered(object):

    """
    Data Handler that creates uniderectional MNIST dataset on the fly.
    Each video contains motion in only one direction         
    Digits starting centered                     
    Moving in 4 axis 
    """
    def __init__(self, train, data_root, seq_len=3, num_digits=1, image_size=64, step = [11, 5]):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size / 2)
        self.seed_is_set = False 
        self.channels = 1
        self.steps = step
        self.directions = ['up', 'down', 'left', 'right','up-right','up-left', 'down-right','down-left']
        self.data = datasets.MNIST(
                    path,
                    train=train,
                    download=True,
                    transform=transforms.Compose(
                    [transforms.Resize(self.digit_size),
                     transforms.ToTensor()]))
        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def set_initial_position(self, direction):
        sx = sy = (self.image_size - self.digit_size) // 2
        return sx, sy

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            digit_steps = self.steps[:]
            
            #sample direction
            direction = np.random.choice(self.directions)
            
            #select initial location based on direction
            sx, sy = self.set_initial_position(direction)
            
            #place digit at selected initial location
            x[0, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
            
            for t in range(1, self.seq_len):
                np.random.shuffle(digit_steps)
                step = digit_steps.pop()
                if direction == 'up':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :step, :, 0] = 0
                elif direction == 'down':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, -step:, :, 0] = 0
                elif direction == 'right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 1)
                    x[t, :, :step, 0] = 0
                elif direction == 'left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 1)
                    x[t, :, -step:, 0] = 0
                elif direction == 'up-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], step , axis = 1)
                    x[t, :step, :, 0] = 0
                    x[t, :, :step, 0] = 0
                elif direction == 'up-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -step , axis = 1)
                    x[t, :step, :, 0] = 0
                    x[t, :, -step:, 0] = 0
                elif direction == 'down-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], step , axis = 1)
                    x[t, -step:, :, 0] = 0
                    x[t, :, :step, 0] = 0
                elif direction == 'down-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -step , axis = 1)
                    x[t, -step:, :, 0] = 0
                    x[t, :, -step:, 0] = 0
        return x

class MovingMNIST_unidir_4_axis_random(object):

    """
    Data Handler that creates uniderectional MNIST dataset on the fly.
    Each video contains motion in only one direction         
    Digits starting randomly positioned                     
    Moving in 4 axis 
    """
    def __init__(self, train, data_root, seq_len=3, num_digits=1, image_size=64, step = [11, 5]):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size / 2)
        self.seed_is_set = False 
        self.channels = 1
        self.steps = step
        self.directions = ['up', 'down', 'left', 'right','up-right','up-left', 'down-right','down-left']
        self.data = datasets.MNIST(
                    path,
                    train=train,
                    download=True,
                    transform=transforms.Compose(
                    [transforms.Resize(self.digit_size),
                     transforms.ToTensor()]))
        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def set_initial_position(self, direction):
        sx = np.random.randint(self.image_size-self.digit_size)
        sy = np.random.randint(self.image_size-self.digit_size)
        return sx, sy

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            digit_steps = self.steps[:]
            
            #sample direction
            direction = np.random.choice(self.directions)
            
            #select initial location based on direction
            sx, sy = self.set_initial_position(direction)
            
            #place digit at selected initial location
            x[0, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
            
            for t in range(1, self.seq_len):
                np.random.shuffle(digit_steps)
                step = digit_steps.pop()
                if direction == 'up':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :step, :, 0] = 0
                elif direction == 'down':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, -step:, :, 0] = 0
                elif direction == 'right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 1)
                    x[t, :, :step, 0] = 0
                elif direction == 'left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 1)
                    x[t, :, -step:, 0] = 0
                elif direction == 'up-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], step , axis = 1)
                    x[t, :step, :, 0] = 0
                    x[t, :, :step, 0] = 0
                elif direction == 'up-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -step , axis = 1)
                    x[t, :step, :, 0] = 0
                    x[t, :, -step:, 0] = 0
                elif direction == 'down-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], step , axis = 1)
                    x[t, -step:, :, 0] = 0
                    x[t, :, :step, 0] = 0
                elif direction == 'down-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -step , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -step , axis = 1)
                    x[t, -step:, :, 0] = 0
                    x[t, :, -step:, 0] = 0
        return x


class MovingMNIST_4_axis_random_sample_step(object):

    """
    Data Handler that creates uniderectional MNIST dataset on the fly.
    Each video contains motion in only one direction         
    Digits starting randomly positioned                     
    Moving in 4 axis with random sample step 
    """
    def __init__(self, train, data_root, seq_len=3, num_digits=1, image_size=64, step = 15):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size / 2)
        self.seed_is_set = False 
        self.channels = 1
        self.step = step
        self.directions = ['up', 'down', 'left', 'right','up-right','up-left', 'down-right','down-left']
        self.data = datasets.MNIST(
                    path,
                    train=train,
                    download=True,
                    transform=transforms.Compose(
                    [transforms.Resize(self.digit_size),
                     transforms.ToTensor()]))
        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]
            #digit_steps = self.steps[:]
            
            #sample direction
            direction = np.random.choice(self.directions)
            
            #select initial location randomly
            sx = np.random.randint(self.image_size-self.digit_size)
            sy = np.random.randint(self.image_size-self.digit_size)
            
            #place digit at selected initial location
            x[0, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
            
            #sample x and y step sizes
            dx = np.random.randint(self.step)
            dy = np.random.randint(self.step)
            for t in range(1, self.seq_len):
                if direction == 'up':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], dy , axis = 0)
                    x[t, :dy, :, 0] = 0
                elif direction == 'down':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -dy , axis = 0)
                    x[t, -dy:, :, 0] = 0
                elif direction == 'right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], dx , axis = 1)
                    x[t, :, :dx, 0] = 0
                elif direction == 'left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -dx , axis = 1)
                    x[t, :, -dx:, 0] = 0
                elif direction == 'up-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], dy , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], dx , axis = 1)
                    x[t, :dy, :, 0] = 0
                    x[t, :, :dx, 0] = 0
                elif direction == 'up-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], dy , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -dx , axis = 1)
                    x[t, :dy, :, 0] = 0
                    x[t, :, -dx:, 0] = 0
                elif direction == 'down-right':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -dy , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], dx , axis = 1)
                    x[t, -dy:, :, 0] = 0
                    x[t, :, :dx, 0] = 0
                elif direction == 'down-left':
                    x[t, :, :, 0] = np.roll(x[t-1, :, :, 0], -dy , axis = 0)
                    x[t, :, :, 0] = np.roll(x[t, :, :, 0], -dx , axis = 1)
                    x[t, -dy:, :, 0] = 0
                    x[t, :, -dx:, 0] = 0
        return x