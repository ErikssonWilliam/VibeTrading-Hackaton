## Step 1: Install Python

1. Open Terminal (Press `Cmd + Space`, type "Terminal", press Enter)
2. Copy and paste this command, then press Enter:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Copy and paste this command, then press Enter:
```bash
brew install python
```
4. **Verify that it worked**
   Copy and paste this command, then press Enter:
```bash
python3 --version
```
  You shall now see a version number.

## Step 2: Install Git

1. Open Terminal (Press `Cmd + Space`, type "Terminal", press Enter)
2. Copy and paste this command, then press Enter:
```bash
brew install git
```

3. **Verify it worked:**
     Write this command in the terminal:
   ```bash
   git --version
   ```
  You shall now see a version number.

## Step 3: Install VS Code

1. Open Terminal (Press `Cmd + Space`, type "Terminal", press Enter)

2. Download VS Code from from [VS Code](https://code.visualstudio.com)

3. Open the downloaded .zip file
4. **Verify it worked:**
       Write this command in the terminal:
   ```bash
   code --version
   ```
  You shall now see a version number.

  ## Running the Game (macOS)

### Step 1: Fork and Clone the Repository

1. Go to the hackathon: [https://github.com/ErikssonWilliam/VibeTrading-Hackaton](https://github.com/ErikssonWilliam/VibeTrading-Hackaton)
2. Click the "Fork" button in the top-right corner to create your own copy
3. Open **Terminal** (press `Cmd + Space`, type "Terminal", press Enter)
4. Clone your forked repository (replace `YourUsername` with your actual GitHub username):
```bash
git clone https://github.com/YourUsername/VibeTrading-Hackaton.git
cd VibeTrading-Hackaton
```

### Step 2: Set up a Python Virtual Environment

1. Create the environment:
```bash
python3 -m venv .venv
```
2. Activate the environment:
```bash
source .venv/bin/activate
```
3. Install required Python packages:
```bash
pip install -r requirements.txt
```
[Go back to Main README](../README.md)