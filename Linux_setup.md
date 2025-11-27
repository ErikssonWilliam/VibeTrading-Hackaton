# Prerequisites

## Step 1: Install Python

1. Open the terminal (Ctrl + Alt + T) and write these commands in the terminal:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

2. **Verify it worked**:
```bash
python3 --version
pip3 --version
```

You shall now see version numbers for both.

## Step 2: Install Git

1. Open the terminal (Ctrl + Alt + T) and write these commands in the terminal:
```bash
sudo apt update
sudo apt install git -y
```

2. **Verify it worked**:
```bash
  git --version
```

You shall now see a version number.

## Step 3: Install VS Code

1. Open the terminal (Ctrl + Alt + T) and write this command in the terminal:
```bash
sudo snap install --classic code
```

2. **Verify it worked**:
```bash
  code --version
```

You shall now see a version number.

# Running the game

## Running the Game (Linux)

### Step 1: Fork and Clone the Repository

1. Go to the hackaton [(https://github.com/ErikssonWilliam/VibeTrading-Hackaton)]
2. Click the "Fork" button in the top-right corner to create your own copy
3. Open terminal and clone your forked repository, replace YourUsername with the actual name:
```bash
git clone https://github.com/YourUsername/VibeTrading-Hackaton.git
cd AI-Grand-Prix
```

### Step 2: Setup a python virtual environment
1. Create the environment
```bash
python3 -m venv .venv
```
2. Activate the environment
```bash
source .venv/bin/activate
```
Your terminal prompt should now show (venv) at the beginning

3. Install correct python packages
```bash
pip install -r requirements.txt
```
