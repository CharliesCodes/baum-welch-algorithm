import itertools
import numpy as np
import pandas as pd


class HiddenMarkovModel:

    def __init__(self):
        self.SYMBOLS = [",", "A", "T", "G", "C"]  # V
        self.STATES = [0,1,2] # S

        self.a_matrix = np.zeros((len(self.STATES), len(self.STATES)))
        self.b_matrix = np.zeros((len(self.STATES), len(self.SYMBOLS)))
        #! remove next line later
        self.sequence_output()
        self.alpha = np.zeros((len(self.STATES), len(self.OUTPUT)))
        self.beta = np.zeros((len(self.STATES), len(self.OUTPUT)))

        self.fill_a_matrix()
        self.fill_b_matrix()
        self.sequence_output()

    def fill_a_matrix(self):
        # Transition Probabilities
        self.a_matrix = np.array([
            [0, 0.5, 0.5],
            [0, 0.25, 0.75],
            [1/2, 1/4, 1/4]])

    def fill_b_matrix(self):
        # Emission Probabilities
        self.b_matrix = np.array([
            [1, 0, 0, 0, 0],
            [0, 1/3, 1/3, 1/6, 1/6],
            [0, 1/6, 1/3, 1/6, 1/3]])
        self.b_matrix = pd.DataFrame(self.b_matrix, columns=self.SYMBOLS)

    def sequence_output(self):
        self.OUTPUT = ",CGG,"

    def forward(self):
        self.alpha = np.zeros((len(self.STATES), len(self.OUTPUT)))
        self.alpha[0][0] = 1
        # start at second row (first is initial), start at index 1 because the first row was skipped
        for row_index, row in enumerate(self.alpha.T[1:], 1):
            for cell_index, cell in enumerate(row):
                #* Idea: sum(Last alpha row * fitting A Row) * cell from B
                last_alpha_row = np.copy(self.alpha.T[row_index-1])
                a_matrix_row = np.copy(self.a_matrix.T[cell_index])
                matrix_multiply_result_sum = np.dot(last_alpha_row, a_matrix_row)

                current_nucleo = self.OUTPUT[row_index]
                nucleo_probability = self.b_matrix[current_nucleo][cell_index]
                # 4 ndigits
                cell_result = round(matrix_multiply_result_sum * nucleo_probability, 8)
                self.alpha.T[row_index][cell_index] = cell_result

    def backward(self):
        self.beta = np.zeros((len(self.STATES), len(self.OUTPUT)))
        self.beta[0][-1] = 1
        # start at second row (first is initial), start at index 2 because the last row was skipped and negative starts at -1,
        # [1::-1] means backward loop to second row
        for col_index, col in enumerate(self.beta.T[-2::-1], 2):
            for row_index, val in enumerate(col):

                a_matrix_col = np.copy(self.a_matrix[row_index])

                next_nucleo = self.OUTPUT[-col_index+1]
                b_matrix_col = np.copy(self.b_matrix[next_nucleo].to_numpy())
                next_beta_column = np.copy(self.beta.T[-col_index+1])
                arrays = [a_matrix_col, next_beta_column, b_matrix_col]
                matrix_multiply_result_sum = sum(
                    np.prod(np.vstack(arrays), axis=0))
                # 4 ndigits
                cell_result = round(matrix_multiply_result_sum, 8)
                self.beta.T[-col_index][row_index] = cell_result

    def create_image(self, matrix_name: str, loop: int):
        import os

        WIDTH = 1280
        HEIGHT = 800

        HEIGHT_PARTS = len(self.STATES)
        WIDTH_PARTS = len(self.OUTPUT)

        PART_HEIGHT = HEIGHT / HEIGHT_PARTS
        PART_WIDTH = WIDTH / WIDTH_PARTS

        SMALER = min(PART_HEIGHT, PART_WIDTH)
        CIRCLE_RADIUS = SMALER/4

        MARKER_WIDTH = 10
        MARKER_HIGHT = 7

        x_coords = tuple(round(x, 1)
                        for x in np.linspace(start=PART_WIDTH/2, stop=WIDTH-PART_WIDTH/2, num=WIDTH_PARTS))

        y_coords = tuple(round(y, 1)
                        for y in np.linspace(start=PART_HEIGHT/2, stop=HEIGHT-PART_HEIGHT/2, num=HEIGHT_PARTS))

        coords = list(itertools.product(x_coords, y_coords))

        circles = list(
            f'  <circle cx = "{x}" cy = "{y}" r = "{CIRCLE_RADIUS}" stroke = "black" stroke-width = "3" fill="none"/>' for (x, y) in coords)

        output_text = tuple(
            f'  <text x="{x}" y="20" font-size="2em" text-anchor="middle" alignment-baseline="central">{symbol}</text>' for symbol, x in zip(self.OUTPUT, x_coords))


        if matrix_name == "forward":
            matrix = self.alpha
            arrow = """<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto" markerUnits="strokeWidth"> <polygon points="0 0, 10 3.5, 0 7" /> </marker></defs>"""
            line_settings = 'stroke="#000" stroke-width="2" marker-end="url(#arrow)"'
            lines = tuple(f' <line x1="{x+CIRCLE_RADIUS}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS-MARKER_WIDTH*2}" y2="{y}" {line_settings} />' for (x, y) in coords[:-len(self.STATES)])
            down_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS-MARKER_HIGHT}" y2="{y+PART_HEIGHT-MARKER_WIDTH*2}" {line_settings}/>' for index,
                               (x, y) in enumerate(coords[:-len(self.STATES)]) if (index+1) % len(self.STATES) != 0)
            full_down_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS-MARKER_HIGHT}" y2="{y+2*PART_HEIGHT-MARKER_WIDTH*2}" {line_settings}/>' for index,
                                    (x, y) in enumerate(coords[:-len(self.STATES)]) if (index+3) % len(self.STATES) == 0)
            up_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS-MARKER_HIGHT}" y2="{y-PART_HEIGHT+MARKER_WIDTH*2}" {line_settings}/>' for index,
                             (x, y) in enumerate(coords[:-len(self.STATES)]) if (index) % len(self.STATES) != 0)
            full_up_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS-MARKER_HIGHT}" y2="{y-2*PART_HEIGHT+MARKER_WIDTH*2}" {line_settings}/>' for index,
                                  (x, y) in enumerate(coords[:-len(self.STATES)]) if (index) % len(self.STATES) == 2)
            circles[-len(self.STATES)] = circles[-len(self.STATES)].replace('fill="none"', 'fill="lightskyblue"')
        elif matrix_name == "backward":
            matrix = self.beta
            arrow = """<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"> <polygon points="10 0, 10 7, 0 3.5" /> </marker></defs>"""
            line_settings = 'stroke="#000" stroke-width="2" marker-start="url(#arrow)"'
            lines = tuple(
                f' <line x1="{x+CIRCLE_RADIUS+MARKER_WIDTH*2}" y1="{y}" x2="{x+PART_WIDTH-CIRCLE_RADIUS}" y2="{y}" {line_settings} />' for (x, y) in coords[:-len(self.STATES)])
            down_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS+MARKER_HIGHT}" y1="{y+MARKER_WIDTH*2}" x2="{x+PART_WIDTH-CIRCLE_RADIUS}" y2="{y+PART_HEIGHT}" {line_settings}/>' for index,
                               (x, y) in enumerate(coords[:-len(self.STATES)]) if (index+1) % len(self.STATES) != 0)
            full_down_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS+MARKER_HIGHT}" y1="{y+MARKER_WIDTH*2}" x2="{x+PART_WIDTH-CIRCLE_RADIUS}" y2="{y+2*PART_HEIGHT}" {line_settings}/>' for index,
                                    (x, y) in enumerate(coords[:-len(self.STATES)]) if (index+3) % len(self.STATES) == 0)
            up_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS+MARKER_HIGHT}" y1="{y-MARKER_WIDTH*2}" x2="{x+PART_WIDTH-CIRCLE_RADIUS}" y2="{y-PART_HEIGHT}" {line_settings}/>' for index,
                             (x, y) in enumerate(coords[:-len(self.STATES)]) if (index) % len(self.STATES) != 0)
            full_up_lines = tuple(f' <line x1="{x+CIRCLE_RADIUS+MARKER_HIGHT}" y1="{y-MARKER_WIDTH*2}" x2="{x+PART_WIDTH-CIRCLE_RADIUS}" y2="{y-2*PART_HEIGHT}" {line_settings}/>' for index,
                                  (x, y) in enumerate(coords[:-len(self.STATES)]) if (index) % len(self.STATES) == 2)
            circles[0] = circles[0].replace('fill="none"', 'fill="lightskyblue"')

        df_values = tuple(matrix.T.flatten())

        texts = tuple(f"""<text text-anchor="middle" alignment-baseline="central">
        <tspan x = "{x}" dy = "{y}">{round(text,5)}</tspan>
        </text>
        """ for text, (x, y) in zip(df_values, coords))

        circle_colors = [(int(255-round(255*x, 0)), int(round(255*x, 0)), 0) for x in df_values]

        for index, (color, circle) in enumerate(zip(circle_colors, circles)):
            circles[index] = circle.replace('fill="none"', f'fill="rgb{color}"')

        lines = '\n'.join(lines)
        down_lines = '\n'.join(down_lines)
        full_down_lines = '\n'.join(full_down_lines)
        up_lines = '\n'.join(up_lines)
        full_up_lines = '\n'.join(full_up_lines)
        output_text = '\n'.join(output_text)
        circles = '\n'.join(circles)
        texts = '\n'.join(texts)


        svg = f"""<svg height="{HEIGHT}" width="{WIDTH}">
        {arrow}
        {output_text}
        {circles}
        {lines}

        {down_lines}

        {full_down_lines}

        {up_lines}

        {full_up_lines}

        {texts}
        </svg>
        """

        current_dir = os.getcwd()
        path = f"{current_dir}/images/{matrix_name}"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/{matrix_name}_{loop}.svg", "w") as text_file:
            text_file.write(svg)

    def recalculate_a(self):
        for row_index, row in enumerate(self.a_matrix):
            for column_index, value in enumerate(row):
                nucleos = self.OUTPUT[1:]
                empty_b_list = []
                for n in nucleos:
                    empty_b_list.append(self.b_matrix[n][column_index])
                b_matrix_values = np.array(empty_b_list)
                arrays = [b_matrix_values, np.copy(
                    self.alpha[row_index][:-1]), np.copy(self.beta[column_index][1:])]

                counter = sum(np.prod(np.vstack(arrays), axis=0)
                                * self.a_matrix[row_index][column_index])
                denominator = np.dot(
                    np.copy(self.alpha[row_index][1:]), np.copy(self.beta[row_index][1:]))

                if (counter != 0) and (denominator != 0):
                    result = counter / denominator
                else:
                    result = 0

                self.a_matrix[row_index][column_index] = round(
                    result, 8)

    def recalculate_b(self):
        for row_index, row in self.b_matrix.iloc[1:].iterrows():
            for column_index, value in enumerate(row):
                # find all indexes from output, where current b matrix column name appears
                all_nucleo_indexes = [i for i, ltr in enumerate(
                    self.OUTPUT) if ltr == self.b_matrix.columns[column_index]]

                counter = np.dot(
                    self.alpha[row_index][all_nucleo_indexes], self.beta[row_index][all_nucleo_indexes])
                denominator = np.dot(
                    self.alpha[row_index], self.beta[row_index])
                if counter != 0 or denominator != 0:
                    result = counter / denominator
                else:
                    result = 0
                self.b_matrix[self.b_matrix.columns[column_index]
                                ].iloc[row_index] = round(result, 8)

    def baum_welch(self):
        max_loops = 100
        loop = 1
        while loop < max_loops:
            last_alpha_result = self.alpha[0][-1]

            self.forward()
            hmm.create_image("forward", loop)

            new_alpha_result = round(self.alpha[0][-1], 3)
            if last_alpha_result == new_alpha_result:
                print(f"Calculation stopped - stable Possibility : {round(new_alpha_result,8)*100}% after {loop} loops!\n")
                return

            self.backward()
            hmm.create_image("backward", loop)

            self.recalculate_a()
            self.recalculate_b()

            loop += 1

    def check(self) -> bool:
        # Check if forward and backword worked
        return hmm.beta[0][0] == round(np.dot(hmm.beta.T[1], hmm.alpha.T[1]), 8)

    def __repr__(self) -> str:
        a_df = pd.DataFrame(
            self.a_matrix, columns=self.STATES)
        b_df = pd.DataFrame(self.b_matrix, columns=self.SYMBOLS)
        alpha_df = pd.DataFrame(self.alpha, columns=[x for x in self.OUTPUT])
        beta_df = pd.DataFrame(self.beta, columns=[x for x in self.OUTPUT])

        rep = "Transition Probabilities - A-Matrix\n" + \
            f"{a_df}\n\n" + "Emission Probabilities - B-Matrix\n" + \
            f"{b_df}" + "\n\nAlpha\n" + f"{alpha_df}" + \
            f"\nBeta\n" + f"{beta_df}"
        return rep



if __name__ == '__main__':
    hmm = HiddenMarkovModel()
    print(hmm)

    print("\n\n =========== BAUM-WELCH START ==============\n")
    hmm.baum_welch()
    print(hmm)
