# Google Colab CLI - Practical Remote Development Guide

This guide summarizes a practical workflow for using [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) as a remote development environment from macOS or Linux.

It covers:

- creating CPU/GPU Colab runtimes
- requesting high-memory machines
- listing and inspecting sessions
- attaching to an existing session
- configuring `~/.ssh/config`
- using VS Code Remote-SSH
- mounting Google Drive
- disconnecting vs stopping a runtime
- a recommended daily workflow

> **Important:** Colab runtimes are ephemeral. Anything stored only on the VM can disappear when the runtime is stopped or reclaimed. Use Google Drive or another persistent storage location for important files.

---

## 1. What is `google-colab-cli`?

`google-colab-cli` is Google's command-line interface for working with Colab runtimes directly from a terminal.

It can:

- provision CPU, GPU, and TPU runtimes
- select supported accelerators
- request high-memory machine shapes where supported
- execute scripts and notebooks
- install packages
- upload/download files
- mount Google Drive
- expose the runtime through SSH
- support VS Code Remote-SSH and similar tools
- inspect and stop running sessions

Official repository:

<https://github.com/googlecolab/google-colab-cli>

For this setup, the CLI executable is:

```bash
/Users/keshav/.local/bin/colab
```

You can verify your installation with:

```bash
which colab
```

---

# 2. Basic Session Management

## Create a CPU runtime

```bash
colab new -s vscode-cpu
```

## Create a CPU runtime with high RAM

```bash
colab new -s vscode-cpu-highram --high-mem
```

## Create a GPU runtime

Example with an L4:

```bash
colab new -s vscode-l4 --gpu L4
```

Example with an A100:

```bash
colab new -s vscode-a100 --gpu A100
```

Example with an A100 and high RAM:

```bash
colab new -s vscode-a100-highram --gpu A100 --high-mem
```

---

# 3. Available Compute Options

The current CLI recognizes the following GPU selectors:

| Compute | CLI option | Example |
|---|---|---|
| CPU | none | `colab new -s vscode-cpu` |
| T4 | `--gpu T4` | `colab new -s vscode-t4 --gpu T4` |
| L4 | `--gpu L4` | `colab new -s vscode-l4 --gpu L4` |
| G4 | `--gpu G4` | `colab new -s vscode-g4 --gpu G4` |
| A100 | `--gpu A100` | `colab new -s vscode-a100 --gpu A100` |
| H100 | `--gpu H100` | `colab new -s vscode-h100 --gpu H100` |

The CLI also supports TPU runtimes such as:

```bash
colab new -s vscode-tpu --tpu v5e1
```

and:

```bash
colab new -s vscode-tpu --tpu v6e1
```

## High-memory machines

Add:

```bash
--high-mem
```

when creating a runtime:

```bash
colab new -s vscode-a100-highram --gpu A100 --high-mem
```

High-memory availability depends on your Colab entitlement, quota, accelerator, and current backend capacity.

The official CLI documentation notes that accelerators with only one machine shape, such as L4, can ignore `--high-mem`.

> The fact that a GPU name is accepted by the CLI does not guarantee that your account can allocate it at a particular moment.

---

# 4. List Existing Sessions

Before creating another runtime, check what is already running:

```bash
colab sessions
```

Inspect a specific runtime:

```bash
colab status -s vscode-a100
```

This is useful for checking:

- runtime status
- accelerator
- machine shape
- whether the kernel is idle or busy
- session metadata

---

# 5. Attach to an Existing Session

This is one of the most useful features.

Suppose you previously created:

```bash
colab new -s vscode-a100 --gpu A100
```

and then disconnected without stopping it.

You can attach again with:

```bash
colab ssh -s vscode-a100
```

If you have an SSH alias configured, you can simply use:

```bash
ssh colab-a100
```

As long as the runtime is still alive, this reconnects to the existing session instead of creating a new VM.

## Leave the session without stopping it

Inside SSH:

```bash
exit
```

This only disconnects your terminal.

It does **not** stop the Colab runtime.

You can reconnect later:

```bash
ssh colab-a100
```

---

# 6. Disconnect vs Stop

These are very different operations.

## Disconnect only

```bash
exit
```

Result:

- SSH closes
- Colab VM stays alive
- running jobs may continue
- Google Drive remains mounted
- you can reconnect later

## Stop the runtime

From your local terminal:

```bash
colab stop -s vscode-a100
```

Result:

- runtime is terminated
- ephemeral VM storage is lost
- Drive mount disappears
- you must create a new runtime next time

A useful mental model is:

```text
exit
  =
disconnect

colab stop
  =
destroy/release runtime
```

---

# 7. Recommended SSH Configuration

Instead of typing long `colab ssh` commands, add aliases to:

```bash
~/.ssh/config
```

For this setup, use the full CLI path:

```text
/Users/keshav/.local/bin/colab
```

Using the absolute executable path is helpful because SSH `ProxyCommand` does not always inherit the same shell `PATH` as your interactive terminal.

## Full SSH configuration

```ssh
# ============================================================
# Google Colab - CPU
# ============================================================
Host colab-cpu
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-cpu
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - CPU High RAM
# ============================================================
Host colab-cpu-highram
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-cpu-highram --high-mem
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - T4
# ============================================================
Host colab-t4
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-t4 --gpu T4
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - T4 High RAM
# ============================================================
Host colab-t4-highram
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-t4-highram --gpu T4 --high-mem
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - L4
# ============================================================
Host colab-l4
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-l4 --gpu L4
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - G4
# ============================================================
Host colab-g4
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-g4 --gpu G4
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - G4 High RAM
# ============================================================
Host colab-g4-highram
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-g4-highram --gpu G4 --high-mem
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - A100
# ============================================================
Host colab-a100
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-a100 --gpu A100
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - A100 High RAM
# ============================================================
Host colab-a100-highram
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-a100-highram --gpu A100 --high-mem
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - H100
# ============================================================
Host colab-h100
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-h100 --gpu H100
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null


# ============================================================
# Google Colab - H100 High RAM
# ============================================================
Host colab-h100-highram
    ProxyCommand /Users/keshav/.local/bin/colab ssh --proxy-mode -s vscode-h100-highram --gpu H100 --high-mem
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

---

# 8. Why Use Different Session Names?

Notice that these are separate:

```text
vscode-a100
vscode-a100-highram
```

That is intentional.

The provisioning flags such as:

```text
--gpu A100
--high-mem
```

matter when the runtime is created.

If a runtime with the requested session name already exists, `colab ssh` attaches to that existing session.

Separate names reduce ambiguity between standard and high-memory configurations.

---

# 9. Connect Using SSH

Once your SSH config is saved:

```bash
ssh colab-l4
```

or:

```bash
ssh colab-a100
```

or:

```bash
ssh colab-a100-highram
```

If the corresponding runtime already exists, SSH attaches to it.

If it does not exist, `colab ssh` can create the requested runtime using the accelerator options in the `ProxyCommand`.

For important ML jobs, it is often clearer to provision explicitly first:

```bash
colab new -s vscode-a100 --gpu A100
```

verify:

```bash
colab status -s vscode-a100
```

and then connect:

```bash
ssh colab-a100
```

---

# 10. Mount Google Drive

For CLI-created runtimes, use the CLI's dedicated Drive mount command:

```bash
colab drivemount -s vscode-a100
```

By default, Drive is mounted at:

```text
/content/drive
```

Your normal Google Drive content will usually appear under:

```text
/content/drive/MyDrive
```

Verify after connecting:

```bash
ls /content/drive/MyDrive
```

---

# 11. Important Drive-Mount Workflow

If you are already SSH'd into the runtime, **exit the SSH session first**:

```bash
exit
```

Do **not** run:

```bash
colab stop -s vscode-a100
```

because stopping destroys/releases the runtime.

Once back on your **local Mac terminal**, mount Drive:

```bash
colab drivemount -s vscode-a100
```

Then reconnect:

```bash
ssh colab-a100
```

So the correct sequence is:

```text
Inside Colab SSH
      |
      | exit
      v
Local Mac terminal
      |
      | colab drivemount -s vscode-a100
      v
Drive mounted
      |
      | ssh colab-a100
      v
Reconnect to SAME runtime
```

This keeps the VM alive while adding the Drive mount.

---

# 12. Why Not Call `drive.mount()` From a Normal SSH Python Process?

A script such as:

```python
from google.colab import drive
drive.mount("/content/drive")
```

can fail when launched as a regular Python process from an SSH shell because Colab's Drive authentication depends on Colab-specific credential propagation.

For CLI-managed sessions, prefer:

```bash
colab drivemount -s <session>
```

from your local terminal.

---

# 13. Mount Drive After Creating a Runtime

A clean startup sequence for an A100 is:

```bash
# Create the runtime
colab new -s vscode-a100 --gpu A100

# Mount Google Drive
colab drivemount -s vscode-a100

# Connect
ssh colab-a100
```

Then inside the VM:

```bash
cd /content/drive/MyDrive
```

and verify the accelerator:

```bash
nvidia-smi
```

---

# 14. Mount Drive on an Existing Session

Suppose the runtime is already active.

First check:

```bash
colab sessions
```

If you see:

```text
vscode-a100
```

and you are currently SSH'd into it:

```bash
exit
```

Then, from your local Mac:

```bash
colab drivemount -s vscode-a100
```

Reconnect:

```bash
ssh colab-a100
```

There is no need to recreate or stop the runtime.

---

# 15. VS Code Remote-SSH

Once `~/.ssh/config` is configured, VS Code can use the same aliases.

Install the **Remote - SSH** extension.

Then:

```text
Command Palette
    ->
Remote-SSH: Connect to Host
    ->
colab-a100
```

Other examples:

```text
colab-l4
colab-g4
colab-a100-highram
```

VS Code will use the SSH `ProxyCommand` and open the Colab runtime like a remote Linux development machine.

---

# 16. Ghostty Terminal Compatibility

When connecting from Ghostty, a Colab VM may not know the terminal type:

```text
xterm-ghostty
```

This can produce errors such as:

```text
'xterm-ghostty': unknown terminal type
```

A temporary fix is:

```bash
export TERM=xterm-256color
```

To make it automatic for the lifetime of that VM, add to the remote `~/.bashrc`:

```bash
if [ "$TERM" = "xterm-ghostty" ]; then
    export TERM=xterm-256color
fi
```

Then:

```bash
source ~/.bashrc
```

This persists across SSH reconnects to the same runtime, but a newly created Colab VM will have a fresh filesystem.

---

# 17. Switching From CPU to GPU

An existing CPU runtime is not upgraded in place to a GPU runtime.

For example, if you have:

```bash
colab new -s vscode-cpu
```

you cannot transform that same VM into an A100 runtime.

Instead:

```bash
colab stop -s vscode-cpu
```

then create the desired GPU runtime:

```bash
colab new -s vscode-a100 --gpu A100
```

Remember to persist important files before stopping the old VM.

---

# 18. Useful Day-to-Day Commands

| Goal | Command |
|---|---|
| List sessions | `colab sessions` |
| Inspect session | `colab status -s vscode-a100` |
| Create CPU | `colab new -s vscode-cpu` |
| Create CPU high RAM | `colab new -s vscode-cpu-highram --high-mem` |
| Create T4 | `colab new -s vscode-t4 --gpu T4` |
| Create L4 | `colab new -s vscode-l4 --gpu L4` |
| Create G4 | `colab new -s vscode-g4 --gpu G4` |
| Create A100 | `colab new -s vscode-a100 --gpu A100` |
| Create A100 high RAM | `colab new -s vscode-a100-highram --gpu A100 --high-mem` |
| Create H100 | `colab new -s vscode-h100 --gpu H100` |
| Attach existing session | `colab ssh -s vscode-a100` |
| Attach through SSH alias | `ssh colab-a100` |
| Mount Drive | `colab drivemount -s vscode-a100` |
| Restart kernel | `colab restart-kernel -s vscode-a100` |
| Open existing runtime in browser | `colab url -s vscode-a100 --open` |
| Stop runtime | `colab stop -s vscode-a100` |
| Verify NVIDIA GPU inside VM | `nvidia-smi` |

---

# 19. Recommended Daily Workflow

## Step 1 - Check whether a runtime already exists

```bash
colab sessions
```

## Step 2 - Create one only if needed

For example:

```bash
colab new -s vscode-a100 --gpu A100
```

or:

```bash
colab new -s vscode-a100-highram --gpu A100 --high-mem
```

## Step 3 - Mount Drive

```bash
colab drivemount -s vscode-a100
```

## Step 4 - Connect

```bash
ssh colab-a100
```

## Step 5 - Verify hardware

```bash
nvidia-smi
```

## Step 6 - Work from persistent storage if appropriate

```bash
cd /content/drive/MyDrive
```

## Step 7 - Disconnect without destroying the VM

```bash
exit
```

## Step 8 - Reattach later

```bash
ssh colab-a100
```

## Step 9 - Stop only when you are truly finished

```bash
colab stop -s vscode-a100
```

---

# 20. Recommended Mental Model

```text
CHECK
colab sessions
      |
      v
CREATE IF NEEDED
colab new ...
      |
      v
MOUNT PERSISTENT STORAGE
colab drivemount ...
      |
      v
CONNECT
ssh colab-a100
      |
      v
WORK
VS Code / SSH / training
      |
      v
DISCONNECT
exit
      |
      v
REATTACH LATER
ssh colab-a100
      |
      v
FINISHED FOR REAL
colab stop ...
```

The most important distinction is:

```text
exit
  -> disconnect only
  -> runtime stays alive

colab stop
  -> terminate runtime
  -> ephemeral VM is released
```

---

# 21. Example: Complete A100 Workflow

```bash
# See currently active runtimes
colab sessions

# Create an A100 runtime if needed
colab new -s vscode-a100 --gpu A100

# Inspect what Colab allocated
colab status -s vscode-a100

# Mount Google Drive from the LOCAL machine
colab drivemount -s vscode-a100

# Connect
ssh colab-a100
```

Inside the runtime:

```bash
nvidia-smi

cd /content/drive/MyDrive
```

When temporarily leaving:

```bash
exit
```

Later:

```bash
ssh colab-a100
```

When completely finished:

```bash
colab stop -s vscode-a100
```

---

# 22. Official References

- Main repository:  
  <https://github.com/googlecolab/google-colab-cli>

- Session management:  
  <https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md>

- Automation and Drive mounting:  
  <https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md>

- `colab run`:  
  <https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md>

- SSH and VS Code Remote-SSH:  
  <https://github.com/googlecolab/google-colab-cli/blob/main/docs/06_ssh_access.md>

---

## Quick Reference

```bash
# List sessions
colab sessions

# Create GPU runtime
colab new -s vscode-a100 --gpu A100

# Create high-memory GPU runtime
colab new -s vscode-a100-highram --gpu A100 --high-mem

# Mount Drive
colab drivemount -s vscode-a100

# Attach directly
colab ssh -s vscode-a100

# Attach through ~/.ssh/config
ssh colab-a100

# Leave WITHOUT stopping
exit

# Reattach
ssh colab-a100

# Stop when completely finished
colab stop -s vscode-a100
```
