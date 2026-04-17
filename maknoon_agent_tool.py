import os
import subprocess
import json
import shutil
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool

def get_binary_path() -> str:
    """Dynamically resolves the maknoon binary path."""
    return os.environ.get("MAKNOON_BINARY") or shutil.which("maknoon") or "maknoon"

def _run_maknoon(cmd: List[str], env: Dict[str, str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Helper to run maknoon with standard timeout and environment."""
    full_env = os.environ.copy()
    full_env.update(env)
    # Both flag and env are supported for maximum robustness
    full_env["MAKNOON_JSON"] = "1"
    
    binary = get_binary_path()
    if cmd[0] == "MAKNOON_PLACEHOLDER":
        cmd[0] = binary

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
        timeout=timeout,
        check=False
    )

@tool
def get_maknoon_secret(service_name: str, vault_name: str = "default") -> Dict[str, Any]:
    """Retrieves a secret (username, password, note) from the Maknoon vault."""
    env = {}
    if "MAKNOON_PASSPHRASE" not in os.environ:
        return {"error": "MAKNOON_PASSPHRASE not set in environment"}

    try:
        # Using the explicit --json flag for clarity in logs
        cmd = ["MAKNOON_PLACEHOLDER", "vault", "get", service_name, "--json", "--vault", vault_name]
        result = _run_maknoon(cmd, env)
        
        if result.returncode != 0:
            try:
                return json.loads(result.stderr)
            except json.JSONDecodeError:
                return {"error": result.stderr.strip() or f"Exit code {result.returncode}"}
        
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

@tool
def set_maknoon_secret(
    service_name: str, 
    password: str, 
    username: str = "", 
    note: str = "", 
    vault_name: str = "default"
) -> Dict[str, Any]:
    """Stores or updates a secret in the Maknoon vault."""
    env = {}
    try:
        cmd = [
            "MAKNOON_PLACEHOLDER", "vault", "set", service_name, password,
            "--json", "--vault", vault_name,
            "--user", username, "--note", note
        ]
        result = _run_maknoon(cmd, env)
        
        if result.returncode != 0:
            try:
                return json.loads(result.stderr)
            except json.JSONDecodeError:
                return {"error": result.stderr.strip()}
        
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}

@tool
def decrypt_maknoon_file(file_path: str, private_key_path: Optional[str] = None) -> str:
    """Decrypts a .makn file to a string."""
    env = {}
    if private_key_path:
        env["MAKNOON_PRIVATE_KEY"] = private_key_path

    try:
        cmd = ["MAKNOON_PLACEHOLDER", "decrypt", file_path, "-o", "-", "--quiet"]
        result = _run_maknoon(cmd, env, timeout=30)
        
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def encrypt_maknoon_file(
    input_path: str, 
    output_path: str, 
    public_key_path: Optional[str] = None, 
    compress: bool = False
) -> str:
    """Encrypts a file or directory using Maknoon."""
    env = {}
    if public_key_path:
        env["MAKNOON_PUBLIC_KEY"] = public_key_path
    
    try:
        cmd = ["MAKNOON_PLACEHOLDER", "encrypt", input_path, "-o", output_path, "--quiet"]
        if compress:
            cmd.append("--compress")
            
        result = _run_maknoon(cmd, env, timeout=60)
        
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        
        return f"Successfully encrypted {input_path} to {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_maknoon_services(vault_name: str = "default") -> List[str]:
    """Lists available service names in the specified vault."""
    env = {}
    try:
        cmd = ["MAKNOON_PLACEHOLDER", "vault", "list", "--json", "--vault", vault_name]
        result = _run_maknoon(cmd, env)
        
        if result.returncode != 0:
            try:
                err_data = json.loads(result.stderr)
                return [f"Error: {err_data.get('error')}"]
            except:
                return [f"Error: {result.stderr.strip()}"]
        
        return json.loads(result.stdout)
    except Exception as e:
        return [f"Error: {str(e)}"]

@tool
def list_maknoon_vaults() -> List[str]:
    """Lists the names of all available Maknoon vaults."""
    vault_dir = os.path.expanduser("~/.maknoon/vaults")
    if not os.path.exists(vault_dir):
        return []
    
    try:
        vaults = [f.replace(".db", "") for f in os.listdir(vault_dir) if f.endswith(".db")]
        return sorted(vaults)
    except Exception as e:
        return [f"Error: {str(e)}"]

if __name__ == "__main__":
    print(f"--- Maknoon Agentic Tools (Binary: {get_binary_path()}) ---")
