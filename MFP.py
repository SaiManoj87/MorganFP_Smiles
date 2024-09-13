import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Load SMILES from the uploaded CSV file
def load_smiles_from_csv(file_path, smiles_column):
    df = pd.read_csv(file_path)
    smiles_list = df[smiles_column].dropna().tolist()  # Get the SMILES column and drop NaNs
    
    molecules = []
    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi, sanitize=False)

        # custom sanitisation
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
            Chem.SanitizeFlags.SANITIZE_CLEANUP
        )

        molecules.append(mol)
    return molecules

# Generate Morgan fingerprints for each molecule
def generate_fingerprints(molecules, radius=2, fp_size=2048):
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fingerprints = [fpgen.GetFingerprint(mol) for mol in molecules if mol is not None]
    return fingerprints

# Compute pairwise Dice similarity between fingerprints
def compute_similarity(fingerprints):
    n = len(fingerprints)
    similarity_matrix = [[0] * n for _ in range(n)]  # Initialize an n x n matrix

    for i in range(n):
        for j in range(i, n):
            similarity = DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric matrix
    return similarity_matrix

# Save fingerprints as a CSV file
def save_fingerprints_to_csv(fingerprints, output_path):
    data = []
    for i, fp in enumerate(fingerprints):
        bitstring = fp.ToBitString()  # Convert to a string of bits
        data.append({"Molecule": i+1, "Fingerprint": bitstring})
    
    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df

# Save fingerprints as a TXT file
def save_fingerprints_to_txt(fingerprints, output_path):
    with open(output_path, 'w') as f:
        for i, fp in enumerate(fingerprints):
            bitstring = fp.ToBitString()  # Convert to a string of bits
            f.write(f"Molecule {i+1}: {bitstring}\n")

# Example usage
csv_file_path = '/path/to/your/file1.csv'  # Path to your uploaded CSV file
smiles_column = 'smiles'  # Replace with the actual column name containing SMILES

# Load molecules from the CSV file
molecules = load_smiles_from_csv(csv_file_path, smiles_column)

# Generate fingerprints
fingerprints = generate_fingerprints(molecules)

# Save fingerprints to CSV
csv_output_path = '/path/to/your/file1/fingerprints.csv'
df_fingerprints = save_fingerprints_to_csv(fingerprints, csv_output_path)

# Save fingerprints to TXT
txt_output_path = '/path/to/your/file1/fingerprints.txt'
save_fingerprints_to_txt(fingerprints, txt_output_path)

# Compute pairwise Dice similarities
similarity_matrix = compute_similarity(fingerprints)

# Optionally print the similarity matrix
for row in similarity_matrix:
    print(row)

# DataFrame (df_fingerprints) already contains fingerprints and can be further used
print(df_fingerprints)
