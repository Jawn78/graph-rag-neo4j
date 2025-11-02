import os
import time
import click
from pathlib import Path
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.live import Live
from rich import print as rprint
from . import setup_environment, verify_environment, ServerManager, Config

console = Console()

def format_status(status: Dict) -> Panel:
    """Format server status as a rich table."""
    table = Table(title="Server Status")
    table.add_column("Server", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Port", style="blue")
    table.add_column("GPU Layers", style="magenta")

    for server_type, details in status.items():
        status_icon = "✓" if details["healthy"] else "✗"
        status_color = "green" if details["healthy"] else "red"
        table.add_row(
            server_type.capitalize(),
            f"[{status_color}]{status_icon}[/{status_color}]",
            str(details["port"]),
            str(details["gpu_layers"])
        )
    
    return Panel(table, title="Graph RAG Server Monitor", border_style="blue")

@click.group(name='setup')
def setup_group():
    """Manage Graph RAG environment and servers.
    
    This group contains commands for managing the Graph RAG environment and servers:
    
    \b
    - init:   Initialize or update the environment
    - verify: Check environment configuration
    - start:  Start server components
    - status: Monitor server health
    
    Examples:
    \b
    Initialize environment:
      python -m graph_rag setup init
    
    Start servers:
      python -m graph_rag setup start
    
    Start only chat server:
      python -m graph_rag setup start --chat-only
    """
    pass

@setup_group.command()
@click.option('--force', is_flag=True, help='Force reinstallation of packages')
def init(force):
    """Initialize the Graph RAG environment.
    
    This command:
    - Sets up CUDA environment variables
    - Installs required Python packages
    - Configures server dependencies
    
    Options:
        --force: Reinstall all packages from scratch
    
    Example:
        python -m graph_rag setup init --force
    """
    with console.status("[bold green]Setting up environment...") as status:
        try:
            setup_environment(force=force)
            status.update("[bold green]Verifying installation...")
            env_status = verify_environment()
            
            if all(env_status["dependencies"].values()):
                console.print("[bold green]✓[/] Environment setup complete!")
                console.print("\n[bold]Environment Details:[/]")
                for pkg, installed in env_status["dependencies"].items():
                    status_icon = "✓" if installed else "✗"
                    status_color = "green" if installed else "red"
                    console.print(f"  [{status_color}]{status_icon}[/{status_color}] {pkg}")
            else:
                console.print("[bold red]✗[/] Some dependencies failed to install:")
                for pkg, installed in env_status["dependencies"].items():
                    if not installed:
                        console.print(f"  [red]✗[/] {pkg}")
                raise click.ClickException("Environment setup incomplete")
                
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            raise click.ClickException("Environment setup failed")

@setup_group.command()
def verify():
    """Verify the Graph RAG environment setup.
    
    Checks:
    - CUDA installation and configuration
    - Required Python packages
    - Environment variables
    - Python version compatibility
    
    Example:
        python -m graph_rag setup verify
    """
    with console.status("[bold green]Verifying environment..."):
        status = verify_environment()
        
        # CUDA Status
        cuda_panel = Panel(
            f"""Status: {'[green]✓ Available' if status['cuda']['status'] else '[red]✗ Not Available'}[/]
Details: {status['cuda']['info']}""",
            title="CUDA Configuration",
            border_style="cyan"
        )
        console.print(cuda_panel)
        
        # Dependencies
        deps_table = Table(title="Dependencies")
        deps_table.add_column("Package", style="cyan")
        deps_table.add_column("Status", justify="center")
        
        for pkg, installed in status["dependencies"].items():
            status_icon = "[green]✓[/]" if installed else "[red]✗[/]"
            deps_table.add_row(pkg, status_icon)
        
        console.print(deps_table)
        
        # Environment Info
        env_panel = Panel(
            f"""Python: {status['environment']['python']}
Platform: {status['environment']['platform']}""",
            title="Environment Information",
            border_style="blue"
        )
        console.print(env_panel)

@setup_group.command()
@click.option('--chat-only', is_flag=True, help='Start only the chat server')
@click.option('--embed-only', is_flag=True, help='Start only the embedding server')
def start(chat_only, embed_only):
    """Start the Graph RAG servers.
    
    By default, starts both chat and embedding servers.
    Use --chat-only or --embed-only to start specific servers.
    
    Servers:
    - Chat Server:     Port 8081, GPU layers: 32
    - Embedding:       Port 8080, GPU layers: All
    
    Options:
        --chat-only:   Start only the chat server
        --embed-only:  Start only the embedding server
    
    Examples:
        python -m graph_rag setup start
        python -m graph_rag setup start --chat-only
        python -m graph_rag setup start --embed-only
    """
    if chat_only and embed_only:
        raise click.ClickException("Cannot specify both --chat-only and --embed-only")
        
    with ServerManager() as manager:
        with Live(console=console, refresh_per_second=4) as live:
            try:
                if chat_only:
                    manager.start_chat_server()
                elif embed_only:
                    manager.start_embedding_server()
                else:
                    manager.start_servers()
                    
                while True:
                    status = manager.get_server_status()
                    live.update(format_status(status))
                    time.sleep(1)
            except KeyboardInterrupt:
                console.print("\n[yellow]Shutting down servers...[/]")

@setup_group.command()
def status():
    """Check Graph RAG server status.
    
    Displays:
    - Server health status
    - Port numbers
    - GPU layer allocation
    - Connection status
    
    Example:
        python -m graph_rag setup status
    """
    manager = ServerManager()
    status = manager.get_server_status()
    console.print(format_status(status))

# Export the setup group
__all__ = ['setup_group']