import typer
from typing import Optional, List
from pathlib import Path
import sys
import time
import os

# Configure UTF-8 output for Windows
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich import print as rprint

from .openrouter import OpenRouterClient
from .cache import CacheManager
from .kernel_analyzer import KernelAnalyzer
from .bench import Benchmarker
from .compiler import CUDACompiler
from .correctness import CorrectnessChecker
from .profiler import CUDAProfiler

app = typer.Typer(
    name="rightnow",
    help="RightNow AI CUDA Kernel Optimizer - Accelerate your GPU code with AI",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True
)
console = Console()

# RightNow AI brand colors
BRAND_COLOR = "#00D4FF"  # Cyan
SUCCESS_COLOR = "#00FF88"  # Green
WARNING_COLOR = "#FFB800"  # Orange
ERROR_COLOR = "#FF0066"  # Red


def show_banner():
    """Display RightNow AI banner."""
    banner = """
    ╦═╗╦╔═╗╦ ╦╔╦╗╔╗╔╔═╗╦ ╦  ╔═╗╦
    ╠╦╝║║ ╦╠═╣ ║ ║║║║ ║║║║  ╠═╣║
    ╩╚═╩╚═╝╩ ╩ ╩ ╝╚╝╚═╝╚╩╝  ╩ ╩╩
    """
    panel = Panel(
        Align.center(
            Text(banner, style=f"bold {BRAND_COLOR}"),
            vertical="middle"
        ),
        title="[bold]CUDA Kernel Optimizer[/bold]",
        subtitle="[dim]Powered by RightNow AI[/dim]",
        border_style=BRAND_COLOR,
        padding=(0, 2)
    )
    console.print(panel)


def create_progress():
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(style=BRAND_COLOR),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40, style=BRAND_COLOR, complete_style=SUCCESS_COLOR),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )


@app.command()
def optimize(
    kernel_file: Path = typer.Argument(
        ..., 
        help="Path to CUDA kernel file (.cu or .cuh)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    output_file: Optional[Path] = typer.Option(
        None, 
        "--output", "-o", 
        help="Output file for optimized kernel"
    ),
    max_registers: Optional[int] = typer.Option(
        None, 
        "--max-registers", "-r",
        help="Maximum registers per thread",
        min=16,
        max=255
    ),
    shared_memory_kb: Optional[int] = typer.Option(
        None, 
        "--shared-memory", "-s",
        help="Maximum shared memory in KB",
        min=1,
        max=96
    ),
    target_gpu: Optional[str] = typer.Option(
        None, 
        "--gpu", "-g",
        help="Target GPU architecture (e.g., sm_80, sm_86)"
    ),
    benchmark_iterations: int = typer.Option(
        100, 
        "--iterations", "-i",
        help="Number of benchmark iterations",
        min=10,
        max=10000
    ),
    variants: int = typer.Option(
        3, 
        "--variants", "-v",
        help="Number of optimization variants to generate",
        min=1,
        max=10
    ),
    force_regenerate: bool = typer.Option(
        False, 
        "--force", "-f",
        help="Force regeneration even if cached"
    )
):
    """
    Optimize a CUDA kernel using AI-powered code generation.
    
    Examples:
        rightnow optimize kernel.cu
        rightnow optimize matmul.cu -o matmul_fast.cu
        rightnow optimize kernel.cu --gpu sm_86 --variants 5
    """
    show_banner()
    
    try:
        # Initialize components
        cache_manager = CacheManager()
        
        # Check API key with better UX
        if not cache_manager.has_api_key():
            console.print(Panel(
                "[yellow][KEY] OpenRouter API key not found[/yellow]\n\n"
                "To get your API key:\n"
                "1. Visit [link]https://openrouter.ai[/link]\n"
                "2. Sign up or log in\n"
                "3. Create an API key in your dashboard",
                title="[bold]API Key Required[/bold]",
                border_style="yellow"
            ))
            
            api_key = typer.prompt("\n[KEY] Please enter your OpenRouter API key", hide_input=True)
            cache_manager.save_api_key(api_key)
            console.print(f"\n[{SUCCESS_COLOR}][OK] API key saved successfully![/{SUCCESS_COLOR}]\n")
        
        api_key = cache_manager.get_api_key()
        openrouter_client = OpenRouterClient(api_key)
        
        # Read and analyze kernel
        console.print(f"\n[LOAD] Loading kernel: [cyan]{kernel_file.name}[/cyan]")
        
        with open(kernel_file, 'r') as f:
            original_code = f.read()
        
        with create_progress() as progress:
            # Analysis phase
            task = progress.add_task("[ANALYZE] Analyzing kernel...", total=100)
            
            kernel_analyzer = KernelAnalyzer()
            analysis = kernel_analyzer.analyze_kernel(original_code)
            
            progress.update(task, completed=100)
        
        # Display analysis results
        console.print(Panel(
            f"[bold cyan]Kernel Analysis Results[/bold cyan]\n\n"
            f"[*] Kernel name: [bold]{analysis['kernel_name']}[/bold]\n"
            f"[PARAMS] Parameters: {len(analysis['parameters'])}\n"
            f"[MEMORY] Shared memory: {analysis.get('shared_memory_usage', 'None detected')}\n"
            f"[PATTERNS] Patterns: {', '.join(analysis.get('patterns', ['None detected']))}\n"
            f"[COMPLEX] Complexity: {analysis.get('complexity', 'Unknown')}",
            border_style=BRAND_COLOR,
            padding=(1, 2)
        ))
        
        # Check cache
        if not force_regenerate:
            cached = cache_manager.get_cached_kernel(
                str(kernel_file),
                analysis['kernel_name']
            )
            if cached:
                console.print(f"\n[{SUCCESS_COLOR}][CACHE] Using cached optimized kernel[/{SUCCESS_COLOR}]")
                if output_file:
                    with open(output_file, 'w') as f:
                        f.write(cached['code'])
                    console.print(f"[{SUCCESS_COLOR}][SAVED] Optimized kernel written to {output_file}[/{SUCCESS_COLOR}]")
                return
        
        # Generate optimization variants
        console.print(f"\n[AI] Generating [bold]{variants}[/bold] optimization variants...\n")
        
        constraints = {
            "max_registers": max_registers or 255,
            "shared_memory_kb": shared_memory_kb or 48,
            "target_gpu": target_gpu or "sm_70"
        }
        
        with create_progress() as progress:
            task = progress.add_task("[GENERATE] Generating variants...", total=variants)
            
            generated_kernels = []
            for i in range(variants):
                kernel_candidates = openrouter_client.generate_kernel_optimizations(
                    original_code=original_code,
                    analysis=analysis,
                    constraints=constraints,
                    num_variants=1
                )
                if kernel_candidates:
                    generated_kernels.extend(kernel_candidates)
                progress.update(task, advance=1)
        
        # Compile and test variants
        console.print(f"\n[BUILD] Compiling and testing kernels...\n")
        
        compiler = CUDACompiler()
        valid_kernels = []
        
        with create_progress() as progress:
            task = progress.add_task("[COMPILE] Compiling...", total=len(generated_kernels))
            
            for i, kernel in enumerate(generated_kernels):
                # Convert KernelCandidate to dict
                kernel_dict = {
                    "code": kernel.code,
                    "operation": kernel.operation,
                    "constraints": kernel.constraints,
                    "metadata": kernel.metadata
                }
                
                try:
                    compiled = compiler.compile_kernel(kernel_dict)
                    valid_kernels.append(compiled)
                    console.print(f"  [{SUCCESS_COLOR}][OK] Variant {i+1} compiled successfully[/{SUCCESS_COLOR}]")
                except Exception as e:
                    console.print(f"  [{ERROR_COLOR}][FAIL] Variant {i+1} failed: {str(e)[:50]}...[/{ERROR_COLOR}]")
                
                progress.update(task, advance=1)
        
        if not valid_kernels:
            console.print(Panel(
                f"[{ERROR_COLOR}][ERROR] No valid optimization variants generated[/{ERROR_COLOR}]\n\n"
                "Try:\n"
                "• Generating more variants with --variants\n"
                "• Adjusting constraints (--max-registers, --shared-memory)\n"
                "• Simplifying your kernel code",
                title="[bold]Optimization Failed[/bold]",
                border_style=ERROR_COLOR
            ))
            sys.exit(1)
        
        # Benchmark variants
        console.print(f"\n[BENCH] Benchmarking {len(valid_kernels)} kernels...\n")
        
        benchmarker = Benchmarker(iterations=benchmark_iterations)
        profiler = CUDAProfiler()
        
        results = []
        with create_progress() as progress:
            task = progress.add_task("[BENCHMARK] Benchmarking...", total=len(valid_kernels))
            
            for kernel in valid_kernels:
                bench_result = benchmarker.benchmark_kernel_standalone(kernel, analysis)
                profile_result = profiler.profile_kernel(kernel, {})
                results.append({
                    "kernel": kernel,
                    "benchmark": bench_result,
                    "profile": profile_result
                })
                progress.update(task, advance=1)
        
        # Select best variant
        best = min(results, key=lambda x: x["benchmark"]["avg_time_ms"])
        
        # Display results
        console.print(f"\n[bold {SUCCESS_COLOR}][RESULTS] Optimization Results[/bold {SUCCESS_COLOR}]\n")
        
        table = Table(
            show_header=True,
            header_style=f"bold {BRAND_COLOR}",
            border_style=BRAND_COLOR,
            box=None,
            padding=(0, 2)
        )
        table.add_column("Variant", style="cyan", justify="center")
        table.add_column("Time (ms)", justify="right", style="yellow")
        table.add_column("vs Original", justify="center", style="green")
        table.add_column("Registers", justify="center")
        table.add_column("Occupancy", justify="center", style="magenta")
        
        for i, result in enumerate(results):
            is_best = result == best
            variant_name = f"{'[BEST] ' if is_best else '       '}{i+1}"
            
            speedup = analysis.get('baseline_time', 0) / result["benchmark"]["avg_time_ms"] if analysis.get('baseline_time') else 0
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            
            table.add_row(
                variant_name,
                f"{result['benchmark']['avg_time_ms']:.3f}",
                speedup_str,
                str(result['profile'].get('register_usage', 'N/A')),
                f"{result['profile'].get('occupancy', 0):.1%}"
            )
        
        console.print(table)
        
        # Save best variant
        if output_file:
            with open(output_file, 'w') as f:
                f.write(best['kernel']['code'])
            
            console.print(Panel(
                f"[{SUCCESS_COLOR}][SUCCESS] Optimized kernel saved![/{SUCCESS_COLOR}]\n\n"
                f"[OUTPUT] Output: [bold]{output_file}[/bold]\n"
                f"[PERF] Performance: [bold]{best['benchmark']['avg_time_ms']:.3f} ms[/bold]",
                title="[bold]Success![/bold]",
                border_style=SUCCESS_COLOR
            ))
        else:
            console.print(f"\n[bold][BEST] Best optimized kernel:[/bold]\n")
            syntax = Syntax(
                best['kernel']['code'][:500] + "...",  # Show preview
                "cuda",
                theme="monokai",
                line_numbers=True
            )
            console.print(Panel(syntax, border_style=BRAND_COLOR))
            console.print(f"\n[dim]Use --output to save the full optimized kernel[/dim]")
        
        # Cache result
        cache_manager.cache_kernel(
            str(kernel_file),
            analysis['kernel_name'],
            best['kernel']
        )
        
    except KeyboardInterrupt:
        console.print(f"\n[{WARNING_COLOR}][WARN] Optimization cancelled by user[/{WARNING_COLOR}]")
        sys.exit(0)
    except Exception as e:
        console.print(Panel(
            f"[{ERROR_COLOR}][ERROR] Error: {e}[/{ERROR_COLOR}]",
            title="[bold]Error[/bold]",
            border_style=ERROR_COLOR
        ))
        sys.exit(1)


@app.command()
def analyze(
    kernel_file: Path = typer.Argument(
        ...,
        help="Path to CUDA kernel file",
        exists=True
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed analysis"
    )
):
    """
    Analyze a CUDA kernel and identify optimization opportunities.
    
    Examples:
        rightnow analyze kernel.cu
        rightnow analyze matmul.cu --verbose
    """
    try:
        analyzer = KernelAnalyzer()
        with open(kernel_file, 'r') as f:
            code = f.read()
        
        console.print(f"\n[ANALYZE] Analyzing [cyan]{kernel_file.name}[/cyan]...\n")
        
        with create_progress() as progress:
            task = progress.add_task("[ANALYZE] Analyzing kernel patterns...", total=100)
            analysis = analyzer.analyze_kernel(code)
            progress.update(task, completed=100)
        
        # Display analysis in a beautiful panel
        analysis_text = f"""[bold cyan][INFO] Kernel Information[/bold cyan]
├─ Name: [bold]{analysis['kernel_name']}[/bold]
├─ Parameters: {', '.join(analysis['parameters'])}
└─ Launch bounds: {analysis.get('launch_bounds', 'Not specified')}

[bold cyan][MEMORY] Memory Analysis[/bold cyan]
├─ Shared memory: {analysis.get('shared_memory_usage', 'None')}
├─ Global accesses: R={analysis.get('global_accesses', {}).get('reads', 0)} W={analysis.get('global_accesses', {}).get('writes', 0)}
└─ Coalesced: {'Yes' if analysis.get('global_accesses', {}).get('likely_coalesced') else 'No'}

[bold cyan][PATTERNS] Detected Patterns[/bold cyan]"""
        
        patterns = analysis.get('patterns', [])
        if patterns:
            for pattern in patterns:
                analysis_text += f"\n├─ [+] {pattern}"
        else:
            analysis_text += "\n└─ None detected"
        
        # Add optimization opportunities
        opportunities = analysis.get('optimization_opportunities', [])
        if opportunities:
            analysis_text += f"\n\n[bold cyan][OPTIMIZE] Optimization Opportunities[/bold cyan]"
            for i, opp in enumerate(opportunities):
                is_last = i == len(opportunities) - 1
                analysis_text += f"\n{'└' if is_last else '├'}─ {opp}"
        
        console.print(Panel(
            analysis_text,
            title=f"[bold]Analysis Report: {kernel_file.name}[/bold]",
            border_style=BRAND_COLOR,
            padding=(1, 2)
        ))
        
        if verbose:
            # Show additional metrics
            metrics_text = f"""[bold cyan][METRICS] Detailed Metrics[/bold cyan]
├─ Complexity: {analysis.get('complexity', 'Unknown')}
├─ Arithmetic intensity: {analysis.get('arithmetic_intensity', 0):.2f}
├─ Synchronization barriers: {analysis.get('synchronization', {}).get('syncthreads', 0)}
└─ Atomic operations: {analysis.get('synchronization', {}).get('atomic_ops', 0)}"""
            
            console.print(Panel(
                metrics_text,
                border_style="dim cyan",
                padding=(1, 2)
            ))
        
        # Performance hints
        hints = analysis.get('performance_hints', [])
        if hints:
            console.print(f"\n[{WARNING_COLOR}][HINT] Performance Hints:[/{WARNING_COLOR}]")
            for hint in hints:
                console.print(f"  - {hint}")
        
    except Exception as e:
        console.print(f"[{ERROR_COLOR}][ERROR] Error analyzing kernel: {e}[/{ERROR_COLOR}]")
        sys.exit(1)


@app.command()
def benchmark(
    kernel_file: Path = typer.Argument(..., help="Path to CUDA kernel file"),
    data_size: Optional[str] = typer.Option(None, "--size", "-s", help="Data size (e.g., 1024x1024)"),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Number of iterations"),
    compare_with: Optional[Path] = typer.Option(None, "--compare", "-c", help="Compare with another kernel")
):
    """
    Benchmark a CUDA kernel's performance.
    
    Examples:
        rightnow benchmark kernel.cu
        rightnow benchmark kernel.cu --iterations 1000
        rightnow benchmark new.cu --compare old.cu
    """
    console.print(f"\n[BENCH] Benchmarking [cyan]{kernel_file.name}[/cyan]\n")
    console.print("[dim]This feature is coming soon![/dim]")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    reset_api_key: bool = typer.Option(False, "--reset-api-key", help="Reset OpenRouter API key"),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear all cached kernels")
):
    """
    Manage RightNow CLI configuration and cache.
    """
    cache_manager = CacheManager()
    
    if show:
        config_data = cache_manager.get_config()
        
        config_text = f"""[bold cyan][DIRS] Directories[/bold cyan]
├─ Config: {cache_manager.config_dir}
└─ Cache: {cache_manager.cache_dir}

[bold cyan][API] API Configuration[/bold cyan]
└─ API key: {'[OK] Configured' if config_data.get('openrouter_api_key') else '[X] Not set'}

[bold cyan][CACHE] Cache Statistics[/bold cyan]
└─ Cached kernels: {cache_manager.count_cached_kernels()}"""
        
        console.print(Panel(
            config_text,
            title="[bold]RightNow CLI Configuration[/bold]",
            border_style=BRAND_COLOR,
            padding=(1, 2)
        ))
    
    if reset_api_key:
        console.print(f"\n[{WARNING_COLOR}][KEY] Resetting API key...[/{WARNING_COLOR}]")
        api_key = typer.prompt("\n[KEY] Enter new OpenRouter API key", hide_input=True)
        cache_manager.save_api_key(api_key)
        console.print(f"\n[{SUCCESS_COLOR}][OK] API key updated successfully![/{SUCCESS_COLOR}]")
    
    if clear_cache:
        if typer.confirm("\n[CLEAR] Are you sure you want to clear all cached kernels?"):
            cache_manager.clear_cache()
            console.print(f"\n[{SUCCESS_COLOR}][OK] Cache cleared successfully![/{SUCCESS_COLOR}]")


@app.command()
def version():
    """
    Show RightNow CLI version and info.
    """
    from importlib.metadata import version as get_version
    
    try:
        v = get_version("rightnow-cli")
    except:
        v = "1.0.0 (development)"
    
    info_text = f"""[bold cyan]RightNow CLI[/bold cyan] v{v}

[bold]CUDA Kernel Optimizer[/bold]
Part of the RightNow AI Code Editor ecosystem

[dim]Website: https://rightnowai.co
Documentation: https://docs.rightnowai.co/cli
Support: support@rightnowai.co[/dim]"""
    
    console.print(Panel(
        info_text,
        border_style=BRAND_COLOR,
        padding=(1, 2)
    ))


@app.callback()
def main_callback():
    """
    RightNow CLI - AI-Powered CUDA Kernel Optimizer
    
    Part of the RightNow AI Code Editor ecosystem.
    Visit https://rightnowai.co for more information.
    """
    pass


def main():
    app()


if __name__ == "__main__":
    main()