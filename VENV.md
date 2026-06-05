# VIRTUAL ENVIRONMENT

## 1. Create the virtual environment

Run:

```bat
python -m venv venv
```

This creates a folder named `venv` in your project directory.

You can name it anything:

```bat
python -m venv .venv
```

---

## 2. Activate the venv (Windows)

### Command Prompt:

```bat
venv\Scripts\activate
```

### PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

If activation worked, you’ll see something like:

```text
(venv) C:\Users\YourName\projects\spotify-tracker>
```

> [!IMPORTANT]
> ### PowerShell Execution Policy Fix (if needed)
> 
> If PowerShell blocks activation, run **once**:
> 
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> 
> Then try activating again.

---

## 3. Install packages inside the venv

```bat
pip install requests pandas flask
```

Everything installs **only inside this project**, not globally.

---

## 4. Deactivate the venv

When you’re done:

```bat
deactivate
```

---

## 5. Saving and Reinstalling Dependencies 

To save dependencies:

```bat
pip freeze > requirements.txt
```

To reinstall later:

```bat
pip install -r requirements.txt
```