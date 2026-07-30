import click
from typing import Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from . import setup_environment, verify_environment

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