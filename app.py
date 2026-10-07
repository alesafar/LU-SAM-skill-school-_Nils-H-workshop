import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shiny import App, render, ui, reactive

# UI Definition
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Association Controls"),
        ui.input_slider(
            "strength",
            "Association Strength (|r|):",
            min=0.0,
            max=1.0,
            value=0.65,
            step=0.01,
        ),
        ui.input_radio_buttons(
            "direction",
            "Relationship Direction:",
            {"positive": "Positive (+)", "negative": "Negative (−)"},
            selected="positive",
        ),
        ui.hr(),
        ui.input_slider(
            "n_points",
            "Sample Size (N):",
            min=50,
            max=1000,
            value=300,
            step=50,
        ),
        ui.input_action_button(
            "resample",
            "Generate New Sample",
            class_="btn-primary w-100",
        ),
        ui.markdown(
            """
            ---
            **Note:** Moving the slider preserves the underlying points and smoothly rotates the cloud. Click *Generate New Sample* to randomize the seed.
            """
        ),
        width=320,
    ),
    ui.card(
        ui.card_header("Bivariate Normal Scatter Plot with Trend Line"),
        ui.output_plot("scatter_plot", height="550px"),
    ),
    ui.layout_columns(
        ui.value_box("Target Correlation", ui.output_text("val_target_r")),
        ui.value_box("Sample Correlation", ui.output_text("val_sample_r")),
        ui.value_box("R² (Variance Explained)", ui.output_text("val_r_squared")),
    ),
    title="Correlated Normal Variables Visualizer",
)


# Server Logic
def server(input, output, session):

    # Base uncorrelated normal coordinates (kept constant across slider changes)
    @reactive.calc
    def base_normals():
        # Re-run whenever the resample button is clicked or sample size changes
        input.resample()
        n = input.n_points()
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        return z1, z2

    # Derived correlated data
    @reactive.calc
    def correlated_data():
        z1, z2 = base_normals()
        sign = 1.0 if input.direction() == "positive" else -1.0
        r_target = sign * input.strength()

        x = z1
        # Bivariate normal formula: Y = r*Z1 + sqrt(1 - r^2)*Z2
        y = r_target * z1 + np.sqrt(max(0.0, 1.0 - r_target**2)) * z2

        # Empirical statistics
        res = stats.linregress(x, y)
        return {
            "x": x,
            "y": y,
            "target_r": r_target,
            "sample_r": res.rvalue,
            "slope": res.slope,
            "intercept": res.intercept,
            "r_squared": res.rvalue**2,
        }

    @render.text
    def val_target_r():
        return f"{correlated_data()['target_r']:+.2f}"

    @render.text
    def val_sample_r():
        return f"{correlated_data()['sample_r']:+.3f}"

    @render.text
    def val_r_squared():
        return f"{correlated_data()['r_squared']:.1%}"

    @render.plot
    def scatter_plot():
        data = correlated_data()
        x, y = data["x"], data["y"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        # Reference zero axes
        ax.axhline(0, color="#cbd5e1", linestyle="--", linewidth=1, zorder=1)
        ax.axvline(0, color="#cbd5e1", linestyle="--", linewidth=1, zorder=1)

        # Scatter points
        color = "#2563eb" if data["target_r"] >= 0 else "#dc2626"
        ax.scatter(
            x,
            y,
            alpha=0.6,
            s=45,
            color=color,
            edgecolors="white",
            linewidth=0.5,
            label=f"Data points (N = {len(x)})",
            zorder=2,
        )

        # Fitted regression trend line
        x_vals = np.linspace(-3.8, 3.8, 100)
        y_vals = data["slope"] * x_vals + data["intercept"]
        ax.plot(
            x_vals,
            y_vals,
            color="#0f172a",
            linewidth=2.5,
            label=f"Trend line: y = {data['slope']:.2f}x + {data['intercept']:.2f}",
            zorder=3,
        )

        # Fixed limits to observe cloud compression
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_xlabel("Variable X (Normal)", fontsize=11, labelpad=8)
        ax.set_ylabel("Variable Y (Normal)", fontsize=11, labelpad=8)
        ax.set_title(
            f"Bivariate Normal Distribution (r = {data['sample_r']:+.2f})",
            fontsize=13,
            pad=12,
            fontweight="semibold",
        )

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        plt.tight_layout()
        return fig


app = App(app_ui, server)
