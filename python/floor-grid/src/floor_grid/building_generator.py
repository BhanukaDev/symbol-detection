import random
from collections import deque
from .models import Grid


class BuildingShapeGenerator:
    def __init__(
        self, grid: Grid, door_size=2, min_room_width=5, min_building_area_ratio=0.6
    ):
        self.grid = grid
        self.door_size = door_size
        self.min_room_width = min_room_width
        self.min_building_area_ratio = min_building_area_ratio
        self.total_cells = grid.rows * grid.cols
        self.target_building_cells = int(self.total_cells * min_building_area_ratio)

    def generate_building_shape(self):
        """
        Generate a connected building shape with smooth, rectangular edges.
        Ensures no floating islands and maintains connectivity throughout.
        """
        # Start with a rectangular base in the center
        center_r = self.grid.rows // 2
        center_c = self.grid.cols // 2

        # Initialize all cells as non-building
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.grid.grid[r][c].is_building_cell = False

        # Create initial rectangle
        initial_width = self.min_room_width * 2
        initial_height = self.min_room_width * 2

        start_r = max(0, center_r - initial_height // 2)
        end_r = min(self.grid.rows, center_r + initial_height // 2)
        start_c = max(0, center_c - initial_width // 2)
        end_c = min(self.grid.cols, center_c + initial_width // 2)

        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                self.grid.grid[r][c].is_building_cell = True

        # Grow the building by adding rectangular chunks
        current_cells = self._count_building_cells()
        failed_attempts = 0
        max_failed_attempts = 10

        while (
            current_cells < self.target_building_cells
            and failed_attempts < max_failed_attempts
        ):
            # Try to add a rectangular extension
            if not self._add_rectangular_extension():
                failed_attempts += 1
            else:
                failed_attempts = 0  # Reset on success

            current_cells = self._count_building_cells()

        # Final cleanup to ensure smooth edges while maintaining connectivity
        self._smooth_edges()

        # Center the building shape
        self._center_building()

        return self.grid

    def _center_building(self):
        """
        Shift the building to the center of the grid.
        """
        building_cells = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    building_cells.append((r, c))

        if not building_cells:
            return

        # Calculate bounding box
        min_r = min(r for r, c in building_cells)
        max_r = max(r for r, c in building_cells)
        min_c = min(c for r, c in building_cells)
        max_c = max(c for r, c in building_cells)

        # Calculate centers
        shape_center_r = (min_r + max_r) // 2
        shape_center_c = (min_c + max_c) // 2

        grid_center_r = self.grid.rows // 2
        grid_center_c = self.grid.cols // 2

        # Calculate offset
        dr = grid_center_r - shape_center_r
        dc = grid_center_c - shape_center_c

        if dr == 0 and dc == 0:
            return

        # Clear grid
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                self.grid.grid[r][c].is_building_cell = False

        # Apply offset
        for r, c in building_cells:
            new_r, new_c = r + dr, c + dc
            if self.grid.is_valid_coord(new_r, new_c):
                self.grid.grid[new_r][new_c].is_building_cell = True

    def _add_rectangular_extension(self):
        """
        Add a rectangular extension to the building.
        Extensions are at least min_room_width in size.
        """
        # Find all edge cells (building cells with at least one non-building neighbor)
        edge_cells = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    neighbors = self.grid.find_neighbors(r, c)
                    if any(not n.is_building_cell for n in neighbors):
                        edge_cells.append((r, c))

        if not edge_cells:
            return False

        # Try random edge cells until we find one we can extend
        random.shuffle(edge_cells)

        for r, c in edge_cells:
            # Try each direction
            directions = ["up", "down", "left", "right"]
            random.shuffle(directions)

            for direction in directions:
                if self._try_extend_direction(r, c, direction):
                    return True

        return False

    def _try_extend_direction(self, r, c, direction):
        """
        Try to extend the building in a given direction with a rectangular block.
        Only extends if the result remains connected and is reasonable in size.
        """
        # Determine the extension size (random between min_room_width and min_room_width * 2)
        extension_depth = random.randint(self.min_room_width, self.min_room_width * 2)
        extension_width = random.randint(self.min_room_width, self.min_room_width * 3)

        cells_to_add = []
        cells_to_check = set()

        if direction == "up" or direction == "down":
            # Vertical extension
            dr = -1 if direction == "up" else 1

            # Ensure the extension is anchored to existing building width
            # Find the extent of building cells in this row
            existing_cells = [
                col
                for col in range(self.grid.cols)
                if self.grid.grid[r][col].is_building_cell
            ]
            if not existing_cells:
                return False

            # Keep extension within bounds of current building or slightly beyond
            min_existing = min(existing_cells)
            max_existing = max(existing_cells)
            extent = max_existing - min_existing + 1

            start_c = max(0, min_existing - 1)
            end_c = min(self.grid.cols, max_existing + 2)

            # Check if we can extend
            for depth in range(1, extension_depth + 1):
                new_r = r + dr * depth
                if not self.grid.is_valid_coord(new_r, 0):
                    break

                for check_c in range(start_c, end_c):
                    if not self.grid.is_valid_coord(new_r, check_c):
                        return False
                    if self.grid.grid[new_r][check_c].is_building_cell:
                        continue
                    cells_to_add.append((new_r, check_c))
                    cells_to_check.add((new_r, check_c))

        elif direction == "left" or direction == "right":
            # Horizontal extension
            dc = -1 if direction == "left" else 1

            # Find the extent of building cells in this column
            existing_cells = [
                row
                for row in range(self.grid.rows)
                if self.grid.grid[row][c].is_building_cell
            ]
            if not existing_cells:
                return False

            # Keep extension within bounds of current building
            min_existing = min(existing_cells)
            max_existing = max(existing_cells)

            start_r = max(0, min_existing - 1)
            end_r = min(self.grid.rows, max_existing + 2)

            # Check if we can extend
            for depth in range(1, extension_depth + 1):
                new_c = c + dc * depth
                if not self.grid.is_valid_coord(0, new_c):
                    break

                for check_r in range(start_r, end_r):
                    if not self.grid.is_valid_coord(check_r, new_c):
                        return False
                    if self.grid.grid[check_r][new_c].is_building_cell:
                        continue
                    cells_to_add.append((check_r, new_c))
                    cells_to_check.add((check_r, new_c))

        # If we have valid cells to add and they form a reasonable extension
        if len(cells_to_add) >= self.min_room_width:
            # Temporarily add cells
            for cell_r, cell_c in cells_to_add:
                self.grid.grid[cell_r][cell_c].is_building_cell = True

            # Verify building remains connected
            if self.is_connected():
                return True
            else:
                # Rollback if it creates floating islands
                for cell_r, cell_c in cells_to_add:
                    self.grid.grid[cell_r][cell_c].is_building_cell = False
                return False

        return False

    def _smooth_edges(self):
        """
        Remove jagged edges that are smaller than min_room_width.
        Only removes protrusions that don't disconnect the building.
        """
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            cells_to_check = []

            # Collect all edge cells
            for r in range(self.grid.rows):
                for c in range(self.grid.cols):
                    if self.grid.grid[r][c].is_building_cell:
                        neighbors = self.grid.find_neighbors(r, c)
                        if any(not n.is_building_cell for n in neighbors):
                            cells_to_check.append((r, c))

            # Try to remove protrusions
            for r, c in cells_to_check:
                if not self.grid.grid[r][c].is_building_cell:
                    continue

                remove_cell = False
                # Check if this is a vertical protrusion
                if self._is_vertical_protrusion(r, c):
                    remove_cell = True
                # Check if this is a horizontal protrusion
                elif self._is_horizontal_protrusion(r, c):
                    remove_cell = True

                if remove_cell:
                    # Temporarily remove and check connectivity
                    self.grid.grid[r][c].is_building_cell = False
                    if self.is_connected():
                        changed = True
                    else:
                        # Restore if it disconnects the building
                        self.grid.grid[r][c].is_building_cell = True

    def _is_vertical_protrusion(self, r, c):
        """
        Check if this cell is part of a vertical protrusion smaller than min_room_width.
        """
        # Count continuous building cells to the left and right
        left_count = 0
        for check_c in range(c - 1, -1, -1):
            if self.grid.grid[r][check_c].is_building_cell:
                left_count += 1
            else:
                break

        right_count = 0
        for check_c in range(c + 1, self.grid.cols):
            if self.grid.grid[r][check_c].is_building_cell:
                right_count += 1
            else:
                break

        total_width = left_count + 1 + right_count

        # Check if this is an edge cell
        up_is_building = r > 0 and self.grid.grid[r - 1][c].is_building_cell
        down_is_building = (
            r < self.grid.rows - 1 and self.grid.grid[r + 1][c].is_building_cell
        )

        # If it's an edge and the width is less than min_room_width, it's a protrusion
        if (
            not up_is_building or not down_is_building
        ) and total_width < self.min_room_width:
            return True

        return False

    def _is_horizontal_protrusion(self, r, c):
        """
        Check if this cell is part of a horizontal protrusion smaller than min_room_width.
        """
        # Count continuous building cells up and down
        up_count = 0
        for check_r in range(r - 1, -1, -1):
            if self.grid.grid[check_r][c].is_building_cell:
                up_count += 1
            else:
                break

        down_count = 0
        for check_r in range(r + 1, self.grid.rows):
            if self.grid.grid[check_r][c].is_building_cell:
                down_count += 1
            else:
                break

        total_height = up_count + 1 + down_count

        # Check if this is an edge cell
        left_is_building = c > 0 and self.grid.grid[r][c - 1].is_building_cell
        right_is_building = (
            c < self.grid.cols - 1 and self.grid.grid[r][c + 1].is_building_cell
        )

        # If it's an edge and the height is less than min_room_width, it's a protrusion
        if (
            not left_is_building or not right_is_building
        ) and total_height < self.min_room_width:
            return True

        return False

    def _count_building_cells(self):
        """Count the number of building cells."""
        return sum(
            1
            for r in range(self.grid.rows)
            for c in range(self.grid.cols)
            if self.grid.grid[r][c].is_building_cell
        )

    def is_connected(self):
        """
        Check if all building cells form a connected shape (no floating islands).
        Uses BFS to verify connectivity.
        """
        # Find first building cell
        start_cell = None
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    start_cell = (r, c)
                    break
            if start_cell:
                break

        if not start_cell:
            return False

        # BFS to find all connected building cells
        visited = set()
        queue = deque([start_cell])
        visited.add(start_cell)

        while queue:
            r, c = queue.popleft()
            neighbors = self.grid.find_neighbors(r, c)

            for neighbor in neighbors:
                coord = (neighbor.posy, neighbor.posx)
                if neighbor.is_building_cell and coord not in visited:
                    visited.add(coord)
                    queue.append(coord)

        # Count total building cells
        total_building = self._count_building_cells()

        return len(visited) == total_building

    def display_building_shape(self):
        """
        Display the building shape in ASCII.
        ■ = building cell
        □ = non-building cell
        """
        for r in range(self.grid.rows):
            row_display = []
            for c in range(self.grid.cols):
                if self.grid.grid[r][c].is_building_cell:
                    row_display.append("■")
                else:
                    row_display.append("□")
            print(" ".join(row_display))

        # Print statistics
        building_count = self._count_building_cells()
        actual_ratio = building_count / self.total_cells
        print(
            f"\nBuilding cells: {building_count}/{self.total_cells} ({actual_ratio:.2%})"
        )
        print(f"Target ratio: {self.min_building_area_ratio:.2%}")
        print(f"Connected: {self.is_connected()}")
