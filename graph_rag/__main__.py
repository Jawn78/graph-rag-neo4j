"""Command-line interface for graph_rag."""
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from .setup.cli import setup_group
from .setup.server import ServerManager

console = Console()

@click.group()
def cli():
    """Graph RAG CLI - Manage your Graph-based RAG system."""
    pass

# Add command groups
cli.add_command(setup_group, name='setup')

@cli.command()
@click.argument('files', type=click.Path(exists=True, path_type=Path), nargs=-1, required=True)
def ingest(files: tuple):
    """Ingest documents into the graph database.

    Examples:
        python -m graph_rag ingest docs/*.pdf             # Ingest all PDFs
        python -m graph_rag ingest report.pdf memo.docx   # Ingest specific files
    """
    from .config import get_driver, detect_embedding_dim
    from .graph.schema import ensure_schema
    from .ingest import ingest_files
    from .utils.logging import configure_logging

    configure_logging()
    with console.status("[bold green]Ingesting documents..."):
        try:
            driver = get_driver()
            ensure_schema(driver, detect_embedding_dim())
            n_docs, n_chunks = ingest_files(driver, [str(f) for f in files])
        except Exception as e:
            console.print(f"[red]Error during ingestion:[/] {e}")
            raise SystemExit(1)

    console.print(Panel.fit(
        f"[green]✓[/] Successfully ingested documents:\n"
        f"  • Documents: {n_docs}\n"
        f"  • Chunks with embeddings: {n_chunks}",
        title="Ingestion Complete",
        border_style="green"
    ))

@cli.command()
@click.option('--question', '-q', required=True, help='The question to answer')
@click.option('--max-results', type=int, default=6, help='Maximum number of relevant chunks to retrieve')
@click.option('--show-sources', is_flag=True, help='Show source citations')
@click.option('--use-mcp', is_flag=True, help='Try MCP agent gateway before the local chat model')
def ask(question: str, max_results: int, show_sources: bool, use_mcp: bool):
    """Ask a question using the RAG system.

    Examples:
        python -m graph_rag ask -q "What is RAG?"
        python -m graph_rag ask -q "Explain the architecture" --show-sources
        python -m graph_rag ask -q "List main features" --max-results 10
    """
    from .config import get_driver
    from .qa import ask as answer_question
    from .utils.logging import configure_logging

    configure_logging()
    with console.status("[bold green]Thinking..."):
        driver = get_driver()
        result = answer_question(driver, question, top_k=max_results, use_mcp=use_mcp)

    if result.get("error"):
        console.print(f"[red]Error:[/] {result['error']}")
        raise SystemExit(1)

    console.print("\n[bold cyan]Answer:[/]")
    console.print(Markdown(result["answer"] or "(no answer)"))

    if show_sources and result.get("citations"):
        console.print("\n[bold]Sources:[/]")
        for i, source in enumerate(result["citations"], 1):
            console.print(f"{i}. {source}")

@cli.command()
def check():
    """Check the status of all Graph RAG components."""
    from .setup.environment import verify_environment
    
    with console.status("[bold green]Checking Graph RAG system..."):
        # Check environment
        env_status = verify_environment()
        
        # Check servers
        manager = ServerManager()
        server_status = manager.get_server_status()
        
        # Check Neo4j
        try:
            from .graph import check_neo4j_connection
            neo4j_status = check_neo4j_connection()
        except ImportError:
            neo4j_status = {"connected": False, "error": "Neo4j module not available"}
        
        # Display results
        console.print("\n[bold cyan]System Status[/]")
        
        # Environment Status
        console.print("\n[bold]Environment:[/]")
        status_color = "green" if env_status["cuda"]["status"] else "red"
        console.print(f"CUDA: [{status_color}]{'✓' if env_status['cuda']['status'] else '✗'}[/] ({env_status['cuda']['info']})")
        
        for pkg, installed in env_status["dependencies"].items():
            status_color = "green" if installed else "red"
            console.print(f"{pkg}: [{status_color}]{'✓' if installed else '✗'}[/]")
        
        # Server Status
        console.print("\n[bold]Servers:[/]")
        for server, status in server_status.items():
            status_color = "green" if status["healthy"] else "red"
            console.print(f"{server.capitalize()}: [{status_color}]{'✓' if status['healthy'] else '✗'}[/] (Port: {status['port']}, GPU Layers: {status['gpu_layers']})")
        
        # Neo4j Status
        console.print("\n[bold]Database:[/]")
        status_color = "green" if neo4j_status.get("connected", False) else "red"
        console.print(f"Neo4j: [{status_color}]{'✓' if neo4j_status.get('connected', False) else '✗'}[/]")
        if error := neo4j_status.get("error"):
            console.print(f"[red]Error: {error}[/]")

@cli.command()
@click.argument('command', required=False)
@click.option('--all', is_flag=True, help='Show detailed help for all commands')
def help(command, all):
    """Show help information for Graph RAG commands.
    
    Examples:
        python -m graph_rag help           # Show general help
        python -m graph_rag help check     # Show help for 'check' command
        python -m graph_rag help setup     # Show help for setup commands
        python -m graph_rag help --all     # Show detailed help for all commands
    """
    console = Console()
    
    if all:
        console.print("[bold cyan]Graph RAG - Complete Command Reference[/]\n")
        
        # Main commands
        console.print("[bold]Core Commands:[/]")
        console.print("  [green]ingest[/]   - Ingest documents into the graph database")
        console.print("  [green]ask[/]      - Ask questions using the RAG system")
        console.print("  [green]check[/]    - Check comprehensive system status:")
        console.print("                • CUDA and GPU support")
        console.print("                • Required Python packages")
        console.print("                • Server health (embedding and chat)")
        console.print("                • Neo4j database connection")
        console.print("  [green]help[/]     - Show this help message\n")
        
        # Setup commands
        console.print("[bold]Setup Commands:[/] (prefix with 'setup')")
        console.print("  [green]init[/]     - Initialize environment and install dependencies")
        console.print("  [green]verify[/]   - Verify environment setup and dependencies")
        console.print("  [green]start[/]    - Start the server components")
        console.print("  [green]status[/]   - Check server status\n")
        
        # Options for specific commands
        console.print("[bold]Common Options:[/]")
        console.print("  ingest --chunk-size NUM     Set chunk size for text splitting")
        console.print("  ingest --overlap NUM        Set overlap between chunks")
        console.print("  ask -q, --question TEXT     Specify the question to answer")
        console.print("  ask --max-results NUM       Set max number of chunks to retrieve")
        console.print("  ask --show-sources         Show source documents and context")
        console.print("  setup init --force          Force reinstallation of packages")
        console.print("  setup start --chat-only     Start only the chat server")
        console.print("  setup start --embed-only    Start only the embedding server\n")
        
        # Usage examples
        console.print("[bold]Example Usage:[/]")
        console.print("  python -m graph_rag ingest docs/*.pdf             # Ingest documents")
        console.print("  python -m graph_rag ask -q \"What is RAG?\"        # Ask a question")
        console.print("  python -m graph_rag check                        # Check system status")
        console.print("  python -m graph_rag setup init --force           # Initialize setup")
        console.print("  python -m graph_rag setup start --chat-only")
        return

    ctx = click.get_current_context()
    if command is None:
        # Show main help with custom formatting
        console.print("\n[bold cyan]Graph RAG[/] - Graph-based Retrieval Augmented Generation\n")
        console.print("[bold]Available Commands:[/]")
        console.print("  [green]check[/]    Check system status")
        console.print("  [green]setup[/]    Manage environment and servers")
        console.print("  [green]help[/]     Show this help message\n")
        console.print("Use [yellow]python -m graph_rag help --all[/] for detailed documentation")
        console.print("Or [yellow]python -m graph_rag help COMMAND[/] for help on specific commands")
        return
    
    # Get the command object
    cmd = cli.get_command(ctx, command)
    if cmd is None:
        console.print(f"[red]Error:[/] No such command: {command}")
        console.print("\nAvailable commands:")
        for name in sorted(cli.list_commands(ctx)):
            console.print(f"  {name}")
        return
    
    # Show help for specific command
    console.print(f"\n[bold cyan]Help: {command}[/]\n")
    console.print(cmd.get_help(ctx))

if __name__ == '__main__':
    cli()