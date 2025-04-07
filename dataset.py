import zeep
import numpy as np
import torch  # Import PyTorch for saving the tensor

# Initialize the JHTDB client
client = zeep.Client('http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL')
ArrayOfFloat = client.get_type('ns0:ArrayOfFloat')
ArrayOfArrayOfFloat = client.get_type('ns0:ArrayOfArrayOfFloat')
SpatialInterpolation = client.get_type('ns0:SpatialInterpolation')
TemporalInterpolation = client.get_type('ns0:TemporalInterpolation')

# Replace with your own token
token = "edu.jhu.pha.turbulence.testing-201406"

# Parameters for the query
function_name = "GetVelocityAndPressure"
dataset_name = "isotropic1024coarse"
spatial_interpolation = SpatialInterpolation("Lag4")
temporal_interpolation = TemporalInterpolation("None")

# Update the time steps to cover t=0 to t=10 every 0.1 seconds
time_steps = np.arange(0.0, 10.1, 1.0)  # From 0 to 10 inclusive, step 0.1

# Update the grid size
grid_size = 256

# Regenerate the grid points for the new resolution
x = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
y = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
z = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)

# Generate the point_list in the correct format for GetData_Python
point_list = []
for xi in x:
    for yi in y:
        for zi in z:
            point_list.append([xi, yi, zi])

# Define a function to split the point list into chunks of size 4096
def chunk_points(points, chunk_size=4096):
    for i in range(0, len(points), chunk_size):
        yield points[i:i + chunk_size]

# Update the output tensor to match the new time steps and resolution
N_time = len(time_steps)
data_tensor = np.zeros((N_time, 4, grid_size, grid_size, grid_size), dtype=np.float32)

# Query the database for each time step
for t_idx, time in enumerate(time_steps):
    print(f"Querying data for time step {time}...")
    
    # Initialize an array to store results for this time step
    timestep_result = np.zeros((len(point_list), 4), dtype=np.float32)
    
    # Process points in chunks
    for chunk_idx, chunk in enumerate(chunk_points(point_list)):
        print(f"Processing chunk {chunk_idx + 1}...")
        
        # Convert chunk to JHTDB structures
        x_coor = ArrayOfFloat([p[0] for p in chunk])
        y_coor = ArrayOfFloat([p[1] for p in chunk])
        z_coor = ArrayOfFloat([p[2] for p in chunk])
        point_chunk = ArrayOfArrayOfFloat([x_coor, y_coor, z_coor])
        
        # Query the server for this chunk
        chunk_result = client.service.GetData_Python(
            function_name, token, dataset_name, time,
            spatial_interpolation, temporal_interpolation, point_chunk
        )
        
        # Store the chunk result in the appropriate location
        start_idx = chunk_idx * 4096
        end_idx = start_idx + len(chunk)
        timestep_result[start_idx:end_idx] = np.array(chunk_result).reshape((-1, 4))
    
    # Reshape and store in the tensor
    data_tensor[t_idx] = timestep_result.T.reshape(4, grid_size, grid_size, grid_size)

print(data_tensor.shape)  # Should be (N_time, 4, grid_size, grid_size, grid_size)
print("Data retrieval complete.")

# Save the data_tensor locally in .pt format
output_file = "data_tensor.pt"
torch.save(data_tensor, output_file)
print(f"Data tensor saved to {output_file}")