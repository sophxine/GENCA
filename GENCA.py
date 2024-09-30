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
#(I recommend setting the number of mini-batches and mini-batch to low at the beginning, especially if training from scratch, but not make sure to not set it too low.)
batch_size =6  # Mini-batch size
num_batches=12 # Number of mini batches
integration_time = 0.0001  # Initial time horizon for ODE integration
integration_time_increase_rate=0.000
max_integration_time=2.0

noise_std=0.05 # Noise augmentation, 0.05 is a good value to start on, for realism and high accuracy decrease it to not have as much noise during training, it can help with generalizing and reducing error accumulation during inference.
ODE_method="dopri5"
resolution=50 # The image resolution to train on

state_size = 1  # Not sure if increasing this improves performance


# Learning rate with the ReduceLROnPlateau scheduler
lr = 1e-4 # Initial learning rate
min_lr=1e-6
lr_decrease_rate=1e-6
patience=10 # How many epochs of no improvement in the validation loss before decreasing the learning rate


# Settings
image_folder = "data"  # Folder containing the image dataset
current_model_name = "lenia"  # Name for saving the current model
loaded_model_name = "lenia"  # Name of the model to load (if available)
val_ratio = 0.10  # Percent of the last frames to use for validation

trainingphases = 9999999999999999  # Number of training epochs
save_interval=9999999999999999 # Save every x epoch, if you want to train without visualizing you can save regularly here.

train = True  
visualize = True

# Pygame setting
cell_size = 5
FPS = 260 

# CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Print the device being used

# Initialize pygame
pygame.init()


RESOLUTION_WIDTH, RESOLUTION_HEIGHT = resolution, resolution
WIDTH = RESOLUTION_WIDTH * cell_size
HEIGHT = RESOLUTION_HEIGHT * cell_size
GRID_WIDTH = RESOLUTION_WIDTH
GRID_HEIGHT = RESOLUTION_HEIGHT





# Function to draw on the grid with selected color
def draw_on_grid(mouse_pos, color):  # Provide a default color
    x, y = mouse_pos
    x //= cell_size
    y //= cell_size
    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
        grid[y, x] = color

# Simulation loop 
def simulation_loop(model,state_size):
    running = True
    clear_grid = False
    drawing = False
    drawing_paused = False
    simulation_paused = False
    manual_pause = False
    state = torch.zeros((1, state_size), dtype=torch.float32).to(device) # Initialize state
    selected_color = (1.0, 1.0, 1.0)  # Initialize selected color to white *outside* the loop    
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
                        selected_color = grid[y, x]  # Select color from clicked cell
                        
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                    drawing_paused = False
                    if not manual_pause:
                        simulation_paused = False
                        
            if event.type == pygame.MOUSEMOTION and drawing:
                draw_on_grid(pygame.mouse.get_pos(), selected_color)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation_paused = not simulation_paused
                    manual_pause = simulation_paused  # Update manual pause state

                if event.key == pygame.K_c:
                    grid.fill((0.0, 0.0, 0.0))  # Fill with black
                    clear_grid = True

                if event.key == pygame.K_TAB:
                    grid[:] = np.random.rand(GRID_HEIGHT, GRID_WIDTH, 3)

                if event.key == pygame.K_s:
                    torch.save(model.state_dict(), f"{current_model_name}.pt")  # Save model weights
                    print("Model weights saved.")

                # Initialize grid with the first state from the dataset (can be adapted)
                if event.key == pygame.K_i:
                    grid[:] = initial_states[0].copy()

        if clear_grid:
            screen.fill((255, 255, 255))  # White background
            clear_grid = False
        else:


            if not drawing_paused and not simulation_paused:
                grid_tensor = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                predicted_state = model(grid_tensor, state, integration_time).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # Pass integration_time
                grid[:] = predicted_state


            screen.fill((255, 255, 255))  # White background
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    cell_color = (int(grid[y, x, 0] * 255), int(grid[y, x, 1] * 255), int(grid[y, x, 2] * 255))
                    pygame.draw.rect(screen, cell_color, (x * cell_size, y * cell_size, cell_size, cell_size))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()



def train_model(model, train_loader, val_loader, epochs, state_size, 
                integration_time=integration_time,
                noise_std=noise_std):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decrease_rate, patience=patience, min_lr=min_lr) 
    criterion = nn.MSELoss()


    # Training loop

    for epoch in range(epochs):
        epoch_loss = 0.0
        val_loss = 0.0

        if (epoch % save_interval == 0 and epoch != 0):
            torch.save(model.state_dict(), f"{current_model_name}.pt")
            print("saved model")

        if (integration_time <= max_integration_time):
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

            state = torch.zeros((batch_x.shape[0], state_size * 3), dtype=torch.float32).to(device)
            
            optimizer.zero_grad()

            state = odeint(model.ode_func, state,
                           torch.tensor([0.0, integration_time]).to(device), # Ensure time tensor is on the correct device
                           method=ODE_method).to(device)[-1]

            state_short_expanded = state[:, :state_size].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
            state_medium_expanded = state[:, state_size:2 * state_size].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
            state_long_expanded = state[:, 2 * state_size:].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
            
            x = torch.cat((batch_x_noisy, state_short_expanded, state_medium_expanded, state_long_expanded), dim=1)
            outputs = model.conv_layers(x)

            loss = criterion(outputs, batch_y)
            error = batch_y - outputs

            state_update_short = model.state_update_short(error.reshape(batch_x_noisy.shape[0], -1))
            state_update_medium = model.state_update_medium(error.reshape(batch_x_noisy.shape[0], -1))
            state_update_long = model.state_update_long(error.reshape(batch_x_noisy.shape[0], -1))

            # Update the state
            state_short = state_update_short
            state_medium = state_update_medium
            state_long = state_update_long
            state = torch.cat([state_short, state_medium, state_long], dim=1)

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

                state = torch.zeros((batch_x.shape[0], state_size * 3), dtype=torch.float32).to(device)

                state = odeint(model.ode_func, state,
                               torch.tensor([0.0, integration_time]).to(device),
                               method=ODE_method).to(device)[-1]

                state_short_expanded = state[:, :state_size].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
                state_medium_expanded = state[:, state_size:2 * state_size].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
                state_long_expanded = state[:, 2 * state_size:].view(-1, state_size, 1, 1).expand(-1, state_size, batch_x_noisy.shape[2], batch_x_noisy.shape[3])
                x = torch.cat((batch_x_noisy, state_short_expanded, state_medium_expanded, state_long_expanded), dim=1)
                outputs = model.conv_layers(x)

                loss = criterion(outputs, batch_y) 
                val_loss += loss.item() * len(batch_x)

        epoch_loss /= len(train_loader.dataset) 
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} completed. Training Loss: {epoch_loss:.10f}, Validation Loss: {val_loss:.10f}") 
        scheduler.step(val_loss)
        
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
    def __init__(self, state_size, grid_width, grid_height):
        super(CellularAutomataModel, self).__init__()
        self.state_size = state_size * 3
        self.grid_width = grid_width
        self.grid_height = grid_height

        def make_conv_layer(in_channels, out_channels, kernel_size, padding, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation).cuda(),
                nn.LayerNorm([out_channels, self.grid_height, self.grid_width]).cuda(),  # Dynamic LayerNorm
                nn.LeakyReLU().cuda()
            )

        self.conv_layers = nn.Sequential(
            make_conv_layer(3 + self.state_size, 128, 5, 2),
            make_conv_layer(128, 256, 3, 1),
            make_conv_layer(256, 256, 3, 2, dilation=2),
            make_conv_layer(256, 128, 3, 1),
            make_conv_layer(128, 128, 3, 4, dilation=4),
            make_conv_layer(128, 64, 3, 1),
            make_conv_layer(64, 64, 3, 2, dilation=2),
            make_conv_layer(64, 32, 3, 1),
            make_conv_layer(32, 32, 3, 4, dilation=4),
            nn.Conv2d(32, 3, kernel_size=1).cuda(),
            nn.Sigmoid().cuda()
        )

        self.state_update_short = nn.Sequential(
            nn.Linear(3 * grid_width * grid_height, 256).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(256, state_size).cuda(),
            nn.Tanh().cuda()
        )
        self.state_update_medium = nn.Sequential(
            nn.Linear(3 * grid_width * grid_height, 256).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(256, state_size).cuda(),
            nn.Tanh().cuda()
        )
        self.state_update_long = nn.Sequential(
            nn.Linear(3 * grid_width * grid_height, 256).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(256, state_size).cuda(),
            nn.Tanh().cuda()
        )


        self.ode_net = nn.Sequential(
            nn.Linear(self.state_size, 256).cuda(),
            nn.ReLU().cuda(),
            nn.Linear(256, self.state_size).cuda()
        )
        self.ode_func = lambda t, y: self.ode_net(y)


    def forward(self, x, state, integration_time):
        # Move time tensor to the same device as the state
        t = torch.tensor([0.0, integration_time]).to(state.device)  # crucial change

        state = odeint(self.ode_func, state, t, method=ODE_method)[-1]

        state_short_expanded = state[:, :self.state_size // 3].view(-1, self.state_size // 3, 1, 1).expand(-1, self.state_size // 3, x.shape[2], x.shape[3])
        state_medium_expanded = state[:, self.state_size // 3:2 * self.state_size // 3].view(-1, self.state_size // 3, 1, 1).expand(-1, self.state_size // 3, x.shape[2], x.shape[3])
        state_long_expanded = state[:, 2 * self.state_size // 3:].view(-1, self.state_size // 3, 1, 1).expand(-1, self.state_size // 3, x.shape[2], x.shape[3])
        x = torch.cat((x, state_short_expanded, state_medium_expanded, state_long_expanded), dim=1)

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
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation set

# Get grid dimensions from the first item in the dataset
grid_width, grid_height = initial_states[0].shape[0], initial_states[0].shape[1]

# Initialize the model
model = CellularAutomataModel(state_size, grid_width, grid_height).to(device)

model = load_pretrained_model(model, f"{loaded_model_name}.pt")

if train:
    training_thread = threading.Thread(
        target=train_model,
        args=(model, train_loader, val_loader, trainingphases, state_size)
    )
    training_thread.start()

if visualize:
    # Create the pygame window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("GENCA")

    clock = pygame.time.Clock()
    # Initialize the grid with the first state from the dataset
    grid = initial_states[0].copy()
    simulation_loop(model, state_size * 3)
elif not train and not visualize:
    print("Neither training nor visualization is enabled.  Exiting.")
elif train and visualize:
    # Initialize the grid with the first state from the dataset
    grid = initial_states[0].copy()
    simulation_loop(model, state_size * 3) 

if train:
    training_thread.join()  # Wait for training to finish before exiting.

print("Program finished.") 
