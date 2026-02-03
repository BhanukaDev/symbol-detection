from typing import Optional, List, Tuple, Set


class Cell:
    def __init__(self, id, posx=None, posy=None, is_building_cell=True):
        self.id = id
        self.posx = posx
        self.posy = posy
        self.is_building_cell = is_building_cell


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = self._create_grid()

    def _create_grid(self):
        # Create a 2D list of Cell objects
        grid = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                cell_id = (r, c)  # Using (row, col) as id for now
                row_cells.append(Cell(cell_id, posx=c, posy=r))  # Added posx and posy
            grid.append(row_cells)
        return grid

    def is_valid_coord(self, r, c):
        # Helper to check if coordinates are within grid bounds
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_cell(self, r, c) -> Optional[Cell]:
        # Get a cell by its row and column
        if self.is_valid_coord(r, c):
            return self.grid[r][c]
        return None  # Or raise an error for invalid coordinates

    def find_neighbors(self, r, c):
        # Returns a list of valid neighbor cell objects (up, down, left, right)
        neighbors = []
        # Define possible movements (dr, dc)
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in movements:
            new_r, new_c = r + dr, c + dc
            if self.is_valid_coord(new_r, new_c):
                neighbors.append(self.get_cell(new_r, new_c))
        return neighbors

    def is_same_row(self, cell1_coords, cell2_coords):
        # Check if two cells are in the same row
        return cell1_coords[0] == cell2_coords[0]

    def is_same_column(self, cell1_coords, cell2_coords):
        # Check if two cells are in the same column
        return cell1_coords[1] == cell2_coords[1]

    def get_cells_in_direction(self, r, c, direction, distance):
        """
        Returns a list of cells in a specified direction from a starting cell
        up to a given distance.

        Args:
            r (int): Row of the starting cell.
            c (int): Column of the starting cell.
            direction (str): 'up', 'down', 'left', or 'right'.
            distance (int): Number of cells to retrieve in the specified direction.

        Returns:
            list: A list of Cell objects.
        """
        result_cells = []
        dr, dc = 0, 0

        if direction == "up":
            dr = -1
        elif direction == "down":
            dr = 1
        elif direction == "left":
            dc = -1
        elif direction == "right":
            dc = 1
        else:
            raise ValueError("Direction must be 'up', 'down', 'left', or 'right'")

        for i in range(
            1, distance + 1
        ):  # Start from 1 to get cells *from* the current cell
            new_r, new_c = r + dr * i, c + dc * i
            if self.is_valid_coord(new_r, new_c):
                result_cells.append(self.get_cell(new_r, new_c))
            else:
                break  # Hit grid boundary
        return result_cells

    def display_grid_ids(self):
        # Simple helper to print the IDs of the cells in the grid
        for r in range(self.rows):
            row_ids = [self.grid[r][c].id for c in range(self.cols)]
            print(row_ids)


class Room:
    def __init__(self, cell_coords):
        """
        cell_coords: list of (r, c) tuples representing cells in this room
        """
        self.cells = set(cell_coords)  # Set of (r, c) coordinates
        self.walls = []  # Will store wall segments later

    def get_bounds(self):
        """Get the bounding box of this room."""
        if not self.cells:
            return None

        min_r = min(r for r, c in self.cells)
        max_r = max(r for r, c in self.cells)
        min_c = min(c for r, c in self.cells)
        max_c = max(c for r, c in self.cells)

        return (min_r, max_r, min_c, max_c)

    def get_width(self):
        """Get the width of the room's bounding box."""
        bounds = self.get_bounds()
        if not bounds:
            return 0
        return bounds[3] - bounds[2] + 1

    def get_height(self):
        """Get the height of the room's bounding box."""
        bounds = self.get_bounds()
        if not bounds:
            return 0
        return bounds[1] - bounds[0] + 1

    def get_area(self):
        """Get the number of cells in this room."""
        return len(self.cells)

    def is_rectangular(self):
        """Check if the room is a perfect rectangle."""
        return self.get_area() == self.get_width() * self.get_height()


class Wall:
    def __init__(self, cells, wall_type, direction):
        """
        cells: list of (r, c) tuples that make up this wall segment
        wall_type: 'internal' or 'external'
        direction: 'horizontal' or 'vertical'
        """
        self.cells = cells
        self.wall_type = wall_type
        self.direction = direction
        self.has_door = False
        self.door_cells = []  # Cells where the door is placed


class Door:
    def __init__(self, cells, room1=None, room2=None, is_entrance=False):
        """
        cells: list of (r, c) tuples where the door is (should be door_size length)
        room1, room2: rooms connected by this door (None if entrance)
        is_entrance: True if this is the main entrance
        """
        self.cells = cells
        self.room1 = room1
        self.room2 = room2
        self.is_entrance = is_entrance
