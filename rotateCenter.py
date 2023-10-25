import os
import numpy as np
from stl import mesh

InputFolder = "D:/archive/members/mark/STLfiles/"
OutputFolder = "D:/archive/members/mark/RotatedCenteredSTLFiles/"

def AABB_rotate_stl(input_filename, output_filename):
      # Load the STL file
    mesh_data = mesh.Mesh.from_file(input_filename)

    # Get the vertices of the mesh
    vertices = mesh_data.vectors.reshape(-1, 3)

    # Calculate the bounding box of the mesh
    min_coords, max_coords = np.min(vertices, axis=0), np.max(vertices, axis=0)
    dimensions = max_coords - min_coords

    # Determine the axis along which the longest dimension lies
    longest_axis = np.argmax(dimensions)

    # Rotate the mesh so that the longest side faces upwards
    if longest_axis == 0:  # X-axis is the longest
        rotation_axis = np.array([0.0, 0.0, 1.0])
        rotation_angle = np.pi / 2
        mesh_data.rotate(rotation_axis, rotation_angle)
        rotation_axis = np.array([0.0, 1.0, -1.0])
        rotation_angle = np.pi
    elif longest_axis == 1:  # Y-axis is the longest
        rotation_axis = np.array([0.0, 1.0, -1.0])
        rotation_angle = np.pi
    else:  # Z-axis is the longest (no rotation needed)
        rotation_axis = np.array([0.0, 0.0, 0.0])
        rotation_angle = 0

    # Rotate the mesh around the selected axis
    mesh_data.rotate(rotation_axis, rotation_angle)

    # Calculate center after rotation
    newVertices = mesh_data.vectors.reshape(-1, 3)
    newMin, newMax = np.min(newVertices, axis=0), np.max(newVertices, axis=0)
    newCenter = (newMax + newMin)/2


    # Calculate the translation needed to recenter the mesh
    translation = -(newCenter)


    # Translate the mesh to recenter it
    mesh_data.x +=translation[0]
    mesh_data.y +=translation[1]
    mesh_data.z +=translation[2]

    # Save the rotated and recentered mesh to the output file
    mesh_data.save(output_filename)

def batch_rotate_stl_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all STL files in the input folder
    stl_files = [f for f in os.listdir(input_folder) if f.endswith('.stl')]

    for stl_file in stl_files:
        input_path = os.path.join(input_folder, stl_file)
        output_path = os.path.join(output_folder, stl_file)
        AABB_rotate_stl(input_path, output_path)
        print(f"Rotated: {stl_file}")
        
batch_rotate_stl_files(InputFolder, OutputFolder)