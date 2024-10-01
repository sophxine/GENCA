import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import threading
import os
import math
from PIL import Image
from torchdiffeq import odeint
from collections import OrderedDict
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Hyperparameters
noise_std = 0.05 # Noise augmentation, 0.05 is a good value to start on, for realism and high accuracy decrease it to not have as much noise during training, it can help with generalizing and reducing error accumulation during inference.
state_size = 1 # Not sure if increasing this improves performance

# Learning rate with the ReduceLROnPlateau scheduler
lr = 0.001 # Initial learning rate
min_lr = 0.00001
lr_decrease_rate = 0.0001
patience = 20 # How many epochs of no improvement in the validation loss before decreasing the learning rate by lr_decrease_rate.


# Model parameters
hidden_size = 10  # Size of the hidden layers
num_layers=5 # Number of convolutional layers
min_channels=20
max_channels=150



# Training settings
image_folder = "lenia3"  # Folder containing the image dataset
current_model_name = "lenia"  # Name for saving the current model
loaded_model_name = "lenia"  # Name of the model to load (if available)
resolution = 50 # The image resolution to train on
#(I recommend setting the mini-batch size to low at the beginning, especially if training from scratch.)
batch_size = 2  # Mini-batch size
num_batches = 12 # Number of mini batches

train = True  
visualize = True
print_interval=1 # Print every x epoch
print_lr=False # Print learning rate

val_ratio = 0.10  # Percent of the last frames to use for validation
training_phases = 9999999999999999  # Number of training epochs
save_interval = 9999999999999999 # Save every x epoch, if you want to train without visualizing you can save regularly here.

 

# ODE settings
use_ode = False 
integration_time = 0.0001  # Initial time horizon for ODE integration
integration_time_increase_rate = 0.000
max_integration_time = 2.0
ODE_method = "dopri5"


# Pygame setting
cell_size = 2
draw_radius=5
FPS = 60 # Lower it to train faster while visualizing

# CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 



# Initialize pygame
pygame.init()

RESOLUTION_WIDTH, RESOLUTION_HEIGHT = resolution, resolution
WIDTH = RESOLUTION_WIDTH * cell_size
HEIGHT = RESOLUTION_HEIGHT * cell_size
GRID_WIDTH = RESOLUTION_WIDTH
GRID_HEIGHT = RESOLUTION_HEIGHT

# Function to draw on the grid
def draw_on_grid(mouse_pos, color_rgb, radius=draw_radius):
    x, y = mouse_pos
    x //= cell_size
    y //= cell_size

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            distance = math.sqrt((i - y)**2 + (j - x)**2)
            if distance <= radius:
                if 0 <= j < GRID_WIDTH and 0 <= i < GRID_HEIGHT:
                    # Convert RGB to normalized floats
                    color_float = (color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)
                    grid[i, j] = color_float  # Assign the normalized color
                    
# Simulation loop 
def simulation_loop(model, state_size):
    running = True
    clear_grid = False
    drawing = False
    drawing_paused = False
    simulation_paused = False
    manual_pause = False
    state = torch.zeros((1, state_size), dtype=torch.float32).to(device) 
    selected_color = (1.0, 1.0, 1.0) # Initialize with white
    selected_color_rgb = (255, 255, 255)  
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button 
                    drawing = True
                    draw_on_grid(pygame.mouse.get_pos(), selected_color)
                    drawing_paused = True
                    simulation_paused = True
                    

                elif event.button == 3:  # Right mouse button
                    x, y = pygame.mouse.get_pos()
                    x //= cell_size
                    y //= cell_size
                    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                        selected_color_rgb = (int(grid[y, x, 0] * 255), 
                                                int(grid[y, x, 1] * 255), 
                                                int(grid[y, x, 2] * 255)) # Store as RGB 
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    drawing_paused = False
                    if not manual_pause:
                        simulation_paused = False
                        


            if event.type == pygame.MOUSEMOTION and drawing:
                draw_on_grid(pygame.mouse.get_pos(), selected_color_rgb)  # Use stored color

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_paused = not simulation_paused
                    manual_pause = simulation_paused  # Update manual pause state


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid[:] = 0.0  # Set all elements to 0.0 (black)
                    selected_color = (1.0, 1.0, 1.0)  # Reset selected color to white 
                    clear_grid = True

                if event.key == pygame.K_TAB:
                    grid[:] = np.random.rand(GRID_HEIGHT, GRID_WIDTH, 3)

                if event.key == pygame.K_s:
                    torch.save(model.state_dict(), f"{current_model_name}.pt")  # Save model weights
                    print("Model weights saved.")

                # Initialize grid with the first state from the dataset
                if event.key == pygame.K_i:
                    grid[:] = initial_states[0].copy()

        if clear_grid:
            screen.fill((255, 255, 255)) 
            clear_grid = False
        else:


            if not drawing_paused and not simulation_paused:
                grid_tensor = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                if use_ode:
                    predicted_state = model(grid_tensor, state, integration_time).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # Pass integration_time
                else:
                    predicted_state = model(grid_tensor, state).squeeze(0).permute(1, 2, 0).cpu().detach().numpy() 
                grid[:] = predicted_state


            screen.fill((255, 255, 255)) 
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    cell_color = (int(grid[y, x, 0] * 255), int(grid[y, x, 1] * 255), int(grid[y, x, 2] * 255))
                    pygame.draw.rect(screen, cell_color, (x * cell_size, y * cell_size, cell_size, cell_size))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def train_model(model, train_loader, val_loader, epochs, state_size):
    global integration_time # Make integration_time a global variable
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_rate, patience=patience, min_lr=min_lr) 
    criterion = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {total_params}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        val_loss = 0.0

        if (epoch % save_interval == 0 and epoch != 0):
            torch.save(model.state_dict(), f"{current_model_name}.pt")
            print("saved model")

        if (integration_time <= max_integration_time and use_ode):
            integration_time += integration_time_increase_rate
            if (integration_time_increase_rate != 0):
                print("current integration time:", integration_time)

        # Training loop
        model.train() 
        
        # Loop for a specified number of batches
        for i in range(num_batches): 
            try:
                batch_x, batch_y = next(iter(train_loader)) 
            except StopIteration:
                # DataLoader is exhausted, start a new epoch
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
                batch_x, batch_y = next(iter(train_loader)) 

            batch_x = batch_x.permute(0, 3, 1, 2).float().to(device)
            batch_y = batch_y.permute(0, 3, 1, 2).float().to(device)

            noise = torch.randn_like(batch_x) * noise_std
            batch_x_noisy = torch.clamp(batch_x + noise, 0.0, 1.0)

            state = torch.zeros((batch_x.shape[0], state_size), dtype=torch.float32).to(device)
            
            optimizer.zero_grad()

            if use_ode:
                state = odeint(model.ode_func, state,
                               torch.tensor([0.0, integration_time]).to(device),
                               method=ODE_method).to(device)[-1]
                state_expanded = state.view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
            else:
                state_expanded = state.view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
            
            x = torch.cat((batch_x_noisy, state_expanded), dim=1)
            outputs = model.conv_layers(x)

            loss = criterion(outputs, batch_y)
            error = batch_y - outputs

            state_update = model.state_update(error.reshape(batch_x_noisy.shape[0], -1))

            state = state_update 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x_noisy)
        # Validation loop 
        model.eval() 
        with torch.no_grad():
            for i in range(len(val_loader)):
                batch_x, batch_y = next(iter(val_loader))
                batch_x = batch_x.permute(0, 3, 1, 2).float().to(device)
                batch_y = batch_y.permute(0, 3, 1, 2).float().to(device)

                noise = torch.randn_like(batch_x) * noise_std
                batch_x_noisy = torch.clamp(batch_x + noise, 0.0, 1.0)

                state = torch.zeros((batch_x.shape[0], state_size), dtype=torch.float32).to(device)

                if use_ode:
                    state = odeint(model.ode_func, state,
                                   torch.tensor([0.0, integration_time]).to(device),
                                   method=ODE_method).to(device)[-1]
                    state_expanded = state.view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
                else:
                    state_expanded = state.view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
                x = torch.cat((batch_x_noisy, state_expanded), dim=1)
                outputs = model.conv_layers(x)

                loss = criterion(outputs, batch_y) 
                val_loss += loss.item() * len(batch_x)

        epoch_loss /= len(train_loader.dataset) 
        val_loss /= len(val_loader.dataset)
        if epoch%print_interval==0 and epoch!=0:
            print(f"Epoch {epoch + 1}/{epochs} completed. Training Loss: {epoch_loss:.10f}, Validation Loss: {val_loss:.10f}") 
        scheduler.step(val_loss)
        if print_lr==True and epoch%print_interval==0 and epoch!=0:
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")         
# Load images from the folder and create the dataset
def load_images_from_folder(folder, max_workers=None):
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    num_images = len(image_files)
    all_images = [None] * num_images  # Pre-allocate the list to maintain order

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, filename in enumerate(image_files):
            filepath = os.path.join(folder, filename)
            future = executor.submit(load_and_process_image, filepath)
            futures[future] = i # Store the index to put the image in the correct place 

        for future in as_completed(futures):
            index = futures[future]
            all_images[index] = future.result()  # Put the image in the right spot in the list

    return all_images


def load_and_process_image(filepath):
    """Helper function to load and process a single image using CUDA."""

    # Using OpenCV with CUDA
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img = cv2.resize(img, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_array = img / 255.0  # Normalize
    return img_array



class CellularAutomataModel(nn.Module):
    def __init__(self, state_size, grid_width, grid_height, hidden_size, num_layers=num_layers, min_channels=min_channels, max_channels=max_channels): 
        super(CellularAutomataModel, self).__init__()
        self.state_size = state_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.min_channels = min_channels
        self.max_channels = max_channels  # Store max_channels

        def make_conv_layer(in_channels, out_channels, kernel_size, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation).to(device),
                nn.LayerNorm([out_channels, self.grid_height, self.grid_width]).to(device),  # Dynamic LayerNorm
                nn.LeakyReLU().to(device)
            )



        conv_layers = []
        in_channels = 3 + self.state_size
        for i in range(self.num_layers):
            if i == 0:
                conv_layers.append(make_conv_layer(in_channels, self.min_channels, 5, 2))  # Use min_channels for the first layer
                in_channels = self.min_channels 
            elif i % 2 == 0:
                out_channels = min(in_channels * 2, self.max_channels)
                conv_layers.append(make_conv_layer(in_channels, out_channels, 3, 2, dilation=2))
                in_channels = out_channels
            else:
                conv_layers.append(make_conv_layer(in_channels, in_channels, 3, 1))  

        conv_layers.append(nn.Conv2d(in_channels, 3, kernel_size=1).to(device))
        conv_layers.append(nn.Sigmoid().to(device))
        self.conv_layers = nn.Sequential(*conv_layers)


        self.state_update = nn.Sequential(
            nn.Linear(3 * grid_width * grid_height, hidden_size).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(hidden_size, state_size).cuda(),
            nn.Tanh().cuda()
        )

        if use_ode:
            self.ode_net = nn.Sequential(
                nn.Linear(self.state_size, hidden_size).cuda(),
                nn.ReLU().cuda(),
                nn.Linear(hidden_size, self.state_size).cuda()
            )
            self.ode_func = lambda t, y: self.ode_net(y)

    def forward(self, x, state, integration_time=None):

        if use_ode and integration_time is not None:
            t = torch.tensor([0.0, integration_time]).to(state.device)
            state = odeint(self.ode_func, state, t, method=ODE_method)[-1]

        state_expanded = state.view(-1, self.state_size, 1, 1).expand(-1, self.state_size, x.shape[2], x.shape[3])
        x = torch.cat((x, state_expanded), dim=1)

        return self.conv_layers(x)

        
def load_pretrained_model(model, path):
    try:
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        if pretrained_dict:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("Loaded compatible pretrained weights.")
        else:
            print("No compatible pretrained weights found.")
        return model
    except FileNotFoundError:
        print("No pretrained model found at specified path.")
        return model
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}.  Check dimensions.")
        return model


class PairDataset(Dataset):
    def __init__(self, training_data, target_data):
        self.training_data = training_data
        self.target_data = target_data

    def __len__(self):
        return len(self.training_data) - 1 

    def __getitem__(self, idx):
        return self.training_data[idx], self.target_data[idx + 1]


# Load initial states from images
initial_states = load_images_from_folder(image_folder)

# Create Dataset and DataLoader 
dataset = PairDataset(np.array(initial_states[:-1]), np.array(initial_states[1:]))



# Split into training and validation sets
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Modify the split to take the last ones for validation
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

# Get grid dimensions from the first item in the dataset
grid_width, grid_height = initial_states[0].shape[0], initial_states[0].shape[1]

# Initialize the model
model = CellularAutomataModel(state_size, grid_width, grid_height, hidden_size).to(device)

model = load_pretrained_model(model, f"{loaded_model_name}.pt")

if train:
    training_thread = threading.Thread(
        target=train_model,
        args=(model, train_loader, val_loader, training_phases, state_size)
    )
    training_thread.start()

if visualize:
    # Create the pygame window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("GENCA")

    clock = pygame.time.Clock()
    # Initialize the grid with the first state from the dataset
    grid = initial_states[0].copy()
        
    # Create a drawing buffer
    drawing_buffer = np.zeros_like(grid) 

    simulation_loop(model, state_size)
elif not train and not visualize:
    print("Neither training nor visualization is enabled.  Exiting.")
elif train and visualize:
    # Initialize the grid with the first state from the dataset
    grid = initial_states[0].copy()
    simulation_loop(model, state_size) 

if train:
    training_thread.join()  # Wait for training to finish before exiting.

print("Program finished.") 
