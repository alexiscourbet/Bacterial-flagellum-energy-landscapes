#!/usr/bin/env python
import numpy as np
from Bio import PDB
import os
import math
import argparse

def parse_pdb(pdb_file):
    """Parse a PDB file and return the lines and atom coordinates"""
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    atom_lines = []
    atom_coords = []
    
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_lines.append(line)
            
            # Extract coordinates
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_coords.append(np.array([x, y, z]))
    
    return lines, atom_lines, np.array(atom_coords)

def get_rotation_matrix_z(angle_degrees):
    """Create a rotation matrix for rotating around the Z axis"""
    # Convert degrees to radians
    theta = np.radians(angle_degrees)
    
    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix

def rotate_structure(atom_lines, atom_coords, rotation_matrix):
    """Apply rotation to all atoms in the structure"""
    rotated_lines = []
    
    for i, line in enumerate(atom_lines):
        # Apply rotation to coordinates
        rotated_coord = rotation_matrix @ atom_coords[i]
        
        # Format the new line with rotated coordinates
        new_line = line[:30] + f"{rotated_coord[0]:8.3f}{rotated_coord[1]:8.3f}{rotated_coord[2]:8.3f}" + line[54:]
        rotated_lines.append(new_line)
    
    return rotated_lines

def concatenate_pdbs(rotor_lines, axle_lines, output_file):
    """Combine rotated rotor and axle PDB lines into a single PDB file"""
    # Filter out TER and END lines, we'll add them back at the end
    rotor_filtered = [line for line in rotor_lines if not (line.startswith('TER') or line.startswith('END'))]
    axle_filtered = [line for line in axle_lines if not (line.startswith('TER') or line.startswith('END'))]
    
    # Write the combined PDB file
    with open(output_file, 'w') as f:
        # Write REMARK with rotation information
        angle_info = output_file.split('_angle_')[1].split('.pdb')[0]
        f.write(f"REMARK Rotor rotated by {angle_info} degrees around Z axis\n")
        
        # Write the rotor structure
        for line in rotor_filtered:
            f.write(line)
        
        # Add a TER line between structures
        f.write("TER\n")
        
        # Write the axle structure
        for line in axle_filtered:
            f.write(line)
        
        # End the file
        f.write("TER\nEND\n")

def generate_rotational_conformers(rotor_pdb, axle_pdb, angle_increment, output_dir):
    """Generate rotor-axle conformers at different rotational angles"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Parse PDB files
    rotor_full_lines, rotor_atom_lines, rotor_coords = parse_pdb(rotor_pdb)
    axle_full_lines, axle_atom_lines, axle_coords = parse_pdb(axle_pdb)
    
    print(f"Parsed {len(rotor_atom_lines)} atoms from rotor PDB")
    print(f"Parsed {len(axle_atom_lines)} atoms from axle PDB")
    
    # Calculate the number of conformers to generate
    num_conformers = int(360 / angle_increment)
    print(f"Generating {num_conformers} conformers with {angle_increment}° rotation increments")
    
    # Create conformers for each angle
    for i in range(num_conformers + 1):  # Add +1 to include the 360° position
        angle = i * angle_increment
        if i == num_conformers:  # Force the last one to be exactly 360°
            angle = 360.0
        angle_formatted = f"{angle:.6f}"
        
        # Create rotation matrix
        rotation_matrix = get_rotation_matrix_z(angle)
        
        # Rotate rotor structure
        rotated_rotor_lines = rotate_structure(rotor_atom_lines, rotor_coords, rotation_matrix)
        
        # Create output filename
        output_file = os.path.join(output_dir, f"rotor_axle_angle_{angle_formatted}.pdb")
        
        # Concatenate the rotated rotor with the fixed axle
        concatenate_pdbs(rotated_rotor_lines, axle_full_lines, output_file)
        
        # Print progress every 10%
        if i % max(1, num_conformers // 10) == 0 or i == num_conformers:
            print(f"Progress: {i+1}/{num_conformers+1} conformers generated ({(i+1)/(num_conformers+1)*100:.1f}%)")
    
    print(f"\nAll {num_conformers} conformers generated in {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate rotational conformers of a rotor-axle system")
    parser.add_argument("rotor_pdb", help="PDB file for rotor component")
    parser.add_argument("axle_pdb", help="PDB file for axle component")
    parser.add_argument("--angle", type=float, default=1.065088757396449704142011834, 
                        help="Rotation angle increment in degrees (default: 1.065088757396449704142011834)")
    parser.add_argument("--output-dir", default="rotational_conformers", 
                        help="Directory for output PDB files (default: rotational_conformers)")
    args = parser.parse_args()
    
    # Display 
    print("\n" + "="*80)
    print("Sampling rotation...".center(80))
    print("="*80 + "\n")
     
    # Generate the conformers
    generate_rotational_conformers(args.rotor_pdb, args.axle_pdb, args.angle, args.output_dir)

if __name__ == "__main__":
    main() 