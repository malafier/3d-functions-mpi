import subprocess
import tkinter as tk
from tkinter import ttk


def input_not_valid(a_val, b_val):
    if not a_val or not b_val:
        return True
    try:
        a, b = float(a_val), float(b_val)
        if a > b:
            return True
        return False
    except ValueError:
        return True


class App:
    def __init__(self):
        self.root = tk.Tk()

        self.root.geometry("1000x800")
        self.root.resizable(False, False)
        self.root.title("Całkowanie numeryczne 3D")

        self.init_scene()

    def init_scene(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(expand=True)

        title_label = ttk.Label(
            frame, text="Całkowanie numeryczne 3D", font=("Helvetica", 16)
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=10)

        frame2 = ttk.Frame(frame)
        frame2.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

        function_label = ttk.Label(frame2, text="Funkcja:")
        function_label.grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
        self.function_dropdown = ttk.Combobox(
            frame2,
            values=["Rosenbrock", "Ricker wavelet", "Schwefel"],
            state="readonly",
        )
        self.function_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.function_dropdown.grid_configure(columnspan=3, sticky="ew")

        self.spinbox_values = {}
        inputs = [
            ("a\u2093", "ax", 2),
            ("a\u1D67", "ay", 2),
            ("b\u2093", "bx", 3),
            ("b\u1D67", "by", 3),
        ]
        for idx, (label_text, label_key, row_number) in enumerate(inputs):
            label = ttk.Label(frame, text=label_text)
            label.grid(row=row_number, column=idx % 2 * 2, sticky=tk.E, padx=5, pady=5)
            spinbox = ttk.Spinbox(frame, from_=-1000, to=1000, increment=0.1, width=5)
            spinbox.grid(
                row=row_number, column=idx % 2 * 2 + 1, sticky=tk.W, padx=5, pady=5
            )
            self.spinbox_values[label_key] = spinbox

        frame3 = ttk.Frame(frame)
        frame3.grid(row=4, column=0, columnspan=4, padx=5, pady=5)

        n_label = ttk.Label(frame3, text="N:")
        n_label.grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
        self.n_entry = ttk.Entry(frame3)
        self.n_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        n_processes_label = ttk.Label(frame3, text="Liczba procesów:")
        n_processes_label.grid(row=1, column=0, sticky=tk.E, padx=5, pady=5)
        self.n_spinbox = ttk.Spinbox(frame3, from_=1, to=12, increment=1, width=18)
        self.n_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        method_label = ttk.Label(frame3, text="Metoda:")
        method_label.grid(row=2, column=0, sticky=tk.E, padx=5, pady=5)
        self.method_dropdown = ttk.Combobox(
            frame3, values=["Trapezoidy", "Monte Carlo"], state="readonly", width=19
        )
        self.method_dropdown.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        submit_button = ttk.Button(frame, text="Licz", command=self.submit)
        submit_button.grid(row=5, column=0, columnspan=4, pady=10)

        sequential_label = ttk.Label(frame, text="Całkowanie sekwencyjne:")
        sequential_label.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        parallel_label = ttk.Label(frame, text="Całkowanie równoległe:")
        parallel_label.grid(row=6, column=2, columnspan=2, padx=5, pady=5)

        self.result_values = []
        for i in range(4):
            result_frame = ttk.LabelFrame(
                frame, text="Wynik" if i < 2 else "Czas liczenia"
            )
            result_frame.grid(
                row=7 + i // 2,
                column=0 if i % 2 == 0 else 2,
                columnspan=2,
                padx=5,
                pady=5,
                sticky="nsew",
            )
            result_frame.columnconfigure(0, weight=1)
            result_frame.rowconfigure(0, weight=1)

            result_value_label = ttk.Label(result_frame, text="0.0", anchor="center")
            result_value_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

            self.result_values.append(result_value_label)

        speedup_label = ttk.Label(frame, text="Uzyskane przyspieszenie:")
        speedup_label.grid(row=9, column=0, columnspan=2, padx=5, pady=15)

        self.speedup_value_label = ttk.Label(frame, text="0", foreground="red")
        self.speedup_value_label.grid(row=9, column=2, columnspan=2, padx=5, pady=15)

        efficiency_label = ttk.Label(frame, text="Uzyskana efektywność:")
        efficiency_label.grid(row=10, column=0, columnspan=2, padx=5, pady=5)

        self.efficiency_value_label = ttk.Label(frame, text="0", foreground="red")
        self.efficiency_value_label.grid(row=10, column=2, columnspan=2, padx=5, pady=5)

    @staticmethod
    def _prepare_args(
        selected_function,
        ax_value,
        bx_value,
        ay_value,
        by_value,
        n_value,
        selected_method,
        is_sequential=True,
    ):
        args = []

        if selected_function == "Rosenbrock":
            args += ["--function", "rosenbrock"]
        elif selected_function == "Ricker wavelet":
            args += ["--function", "ricker_wavelet"]
        elif selected_function == "Schwefel":
            args += ["--function", "schwefel"]

        args += ["--ax", ax_value]
        args += ["--bx", bx_value]
        args += ["--ay", ay_value]
        args += ["--by", by_value]
        args += ["--n", n_value]

        args += ["--mode"]
        if selected_method == "Trapezoidy":
            args += ["SequentialTrapezoid"] if is_sequential else ["ParallelTrapezoid"]
        elif selected_method == "Monte Carlo":
            args += (
                ["SequentialMonteCarlo"] if is_sequential else ["ParallelMonteCarlo"]
            )

        return args

    def submit(self):
        selected_function = self.function_dropdown.get()
        ax_value = self.spinbox_values["ax"].get()
        bx_value = self.spinbox_values["bx"].get()
        ay_value = self.spinbox_values["ay"].get()
        by_value = self.spinbox_values["by"].get()
        n_value = self.n_entry.get()
        n_processes_value = self.n_spinbox.get()
        selected_method = self.method_dropdown.get()

        if input_not_valid(ax_value, bx_value) or input_not_valid(ay_value, by_value):
            self.result_values[0].config(text="Err")
            self.result_values[1].config(text="Err")
            self.result_values[2].config(text="0 s")
            self.result_values[3].config(text="0 s")
            self.speedup_value_label.config(text="Err")
            self.efficiency_value_label.config(text="Err")
            return

        sequential_args = ["python3", "main.py"]
        parallel_args = ["mpirun", "-np", n_processes_value, "python3", "main.py"]

        sequential_args += self._prepare_args(
            selected_function,
            ax_value,
            bx_value,
            ay_value,
            by_value,
            n_value,
            selected_method,
        )
        parallel_args += self._prepare_args(
            selected_function,
            ax_value,
            bx_value,
            ay_value,
            by_value,
            n_value,
            selected_method,
            False,
        )

        sequential_results = self.run_script(sequential_args)
        parallel_results = self.run_script(parallel_args)

        sequential_result, sequential_time = sequential_results.split("\n")
        parallel_result, parallel_time = parallel_results.split("\n")

        self.result_values[0].config(text=sequential_result)
        self.result_values[1].config(text=parallel_result)
        self.result_values[2].config(text=sequential_time + "s")
        self.result_values[3].config(text=parallel_time + "s")

        speedup = float(sequential_time) / float(parallel_time)
        efficiency = speedup / float(n_processes_value)

        self.speedup_value_label.config(text=speedup)
        self.efficiency_value_label.config(text=efficiency)

    @staticmethod
    def run_script(args):
        result = subprocess.run(args, capture_output=True, text=True)
        return result.stdout.strip()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
