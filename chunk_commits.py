import subprocess

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# 1. Reset to the base commit
run_cmd(['git', 'reset', 'bdf71e15e4b0f7d6676008f538d1b073328d3865'])

# 2. Read changed files
with open('changed_files.txt', 'r') as f:
    files = [line.strip() for line in f if line.strip()]

# 3. Chunk into 100-file chunks
chunk_size = 100
chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

# 4. Add and commit chunks
for i, chunk in enumerate(chunks):
    # Add files
    add_cmd = ['git', 'add', '--'] + chunk
    run_cmd(add_cmd)
    
    # Commit
    commit_cmd = ['git', 'commit', '-m', f"chunk {i+1} of {len(chunks)}"]
    run_cmd(commit_cmd)

# 5. Push
run_cmd(['git', 'push', '-f'])
print("Done!")
