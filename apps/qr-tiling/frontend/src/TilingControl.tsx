import { Flex, Button, Input, Checkbox } from "@luxonis/common-fe-components";
import { useState, useEffect } from "react";
import { useDaiConnection } from "@luxonis/depthai-viewer-common";

export type TilingParams = {
    rows: number;
    cols: number;
    overlap: number;
    global_detection: boolean;
    grid_matrix: number[][];
};

interface TilingControlProps {
    initialParams: TilingParams;
}

function getCellColor(index: number): string {
    const hue = (index * 137.508) % 360;
    return `hsl(${hue}, 70%, 60%)`;
}

function createDefaultMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = [];
    let index = 0;
    for (let r = 0; r < rows; r++) {
        const row: number[] = [];
        for (let c = 0; c < cols; c++) {
            row.push(index++);
        }
        matrix.push(row);
    }
    return matrix;
}

function isAdjacentToValue(
    matrix: number[][],
    row: number,
    col: number,
    value: number
): boolean {
    const neighbors = [
        [row - 1, col],
        [row + 1, col],
        [row, col - 1],
        [row, col + 1],
    ];

    for (const [r, c] of neighbors) {
        if (r >= 0 && r < matrix.length && c >= 0 && c < matrix[0].length) {
            if (matrix[r][c] === value) {
                return true;
            }
        }
    }
    return false;
}

interface GridMatrixEditorProps {
    rows: number;
    cols: number;
    matrix: number[][];
    selectedCell: { row: number; col: number } | null;
    onCellClick: (row: number, col: number) => void;
}

function GridMatrixEditor({
    rows,
    cols,
    matrix,
    selectedCell,
    onCellClick,
}: GridMatrixEditorProps) {
    const valueCounts = new Map<number, number>();
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = matrix[r]?.[c] ?? 0;
            valueCounts.set(val, (valueCounts.get(val) ?? 0) + 1);
        }
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {Array.from({ length: rows }).map((_, rowIdx) => (
                <div key={rowIdx} style={{ display: "flex", gap: 2 }}>
                    {Array.from({ length: cols }).map((_, colIdx) => {
                        const cellValue = matrix[rowIdx]?.[colIdx] ?? 0;
                        const isSelected =
                            selectedCell?.row === rowIdx &&
                            selectedCell?.col === colIdx;
                        const isMerged = (valueCounts.get(cellValue) ?? 0) > 1;

                        const color = isMerged
                            ? getCellColor(cellValue)
                            : "#ffffff";
                        const borderColor = isSelected
                            ? isMerged
                                ? color
                                : "#2196F3"
                            : "rgba(0,0,0,0.2)";

                        return (
                            <div
                                key={colIdx}
                                onClick={() => onCellClick(rowIdx, colIdx)}
                                style={{
                                    width: 40,
                                    height: 40,
                                    backgroundColor: color,
                                    cursor: "pointer",
                                    borderRadius: 4,
                                    border: isSelected
                                        ? `3px solid ${borderColor}`
                                        : `1px solid ${borderColor}`,
                                    boxShadow: isSelected
                                        ? `0 0 8px ${borderColor}`
                                        : "none",
                                }}
                            />
                        );
                    })}
                </div>
            ))}
        </div>
    );
}

export function TilingControl({ initialParams }: TilingControlProps) {
    const connection = useDaiConnection();

    // Initialize directly from props
    const [rows, setRows] = useState(initialParams.rows);
    const [cols, setCols] = useState(initialParams.cols);
    const [overlap, setOverlap] = useState(initialParams.overlap);
    const [globalDetection, setGlobalDetection] = useState(initialParams.global_detection);
    const [gridMatrix, setGridMatrix] = useState<number[][]>(
        initialParams.grid_matrix ?? createDefaultMatrix(initialParams.rows, initialParams.cols)
    );
    const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);

    // Track if user changed rows/cols manually
    const [userChangedSize, setUserChangedSize] = useState(false);

    // Reset grid when user changes rows/cols
    useEffect(() => {
        if (userChangedSize) {
            setGridMatrix(createDefaultMatrix(rows, cols));
            setSelectedCell(null);
            setUserChangedSize(false);
        }
    }, [rows, cols, userChangedSize]);

    const handleRowsChange = (newRows: number) => {
        if (newRows !== rows) {
            setUserChangedSize(true);
            setRows(newRows);
        }
    };

    const handleColsChange = (newCols: number) => {
        if (newCols !== cols) {
            setUserChangedSize(true);
            setCols(newCols);
        }
    };

    const handleCellClick = (row: number, col: number) => {
        if (selectedCell === null) {
            setSelectedCell({ row, col });
            return;
        }

        const selectedValue = gridMatrix[selectedCell.row][selectedCell.col];
        const clickedValue = gridMatrix[row][col];

        if (selectedCell.row === row && selectedCell.col === col) {
            setSelectedCell(null);
            return;
        }

        if (clickedValue === selectedValue) {
            setSelectedCell({ row, col });
            return;
        }

        const isAdjacent = isAdjacentToValue(gridMatrix, row, col, selectedValue);

        if (isAdjacent) {
            const newMatrix = gridMatrix.map((r, rIdx) =>
                r.map((c, cIdx) =>
                    rIdx === row && cIdx === col ? selectedValue : c
                )
            );
            setGridMatrix(newMatrix);
        } else {
            setSelectedCell({ row, col });
        }
    };

    const handleUpdate = () => {
        const config = {
            rows,
            cols,
            overlap,
            global_detection: globalDetection,
            grid_matrix: gridMatrix,
        };

        (connection as any).daiConnection?.postToService(
            "Tiling Config Service",
            config
        );
    };

    return (
        <Flex direction="column" gap="md">
            <span style={{ fontSize: 16, fontWeight: "bold" }}>
                Tiling Configuration
            </span>

            <Flex direction="row" gap="sm" alignItems="center">
                <span>Rows:</span>
                <Input
                    type="number"
                    value={rows}
                    onChange={(e) =>
                        handleRowsChange(
                            Math.max(1, Math.min(8, parseInt(e.target.value) || 1))
                        )
                    }
                    style={{ width: 60 }}
                />

                <span>Cols:</span>
                <Input
                    type="number"
                    value={cols}
                    onChange={(e) =>
                        handleColsChange(
                            Math.max(1, Math.min(8, parseInt(e.target.value) || 1))
                        )
                    }
                    style={{ width: 60 }}
                />
            </Flex>

            <Flex direction="row" gap="sm" alignItems="center">
                <span>Overlap:</span>
                <Input
                    type="number"
                    value={overlap}
                    onChange={(e) => setOverlap(parseFloat(e.target.value) || 0)}
                    min={0}
                    max={0.99}
                    step={0.05}
                    style={{ width: 80 }}
                />
            </Flex>

            <Checkbox
                value={globalDetection}
                label="Global Detection (include full image)"
                onChange={() => setGlobalDetection(!globalDetection)}
            />

            <Flex direction="column" gap="sm">
                <GridMatrixEditor
                    rows={rows}
                    cols={cols}
                    matrix={gridMatrix}
                    selectedCell={selectedCell}
                    onCellClick={handleCellClick}
                />

                <Button
                    onClick={() => {
                        setGridMatrix(createDefaultMatrix(rows, cols));
                        setSelectedCell(null);
                    }}
                >
                    Reset Grid
                </Button>
            </Flex>

            <Button onClick={handleUpdate}>Update Tiling</Button>
        </Flex>
    );
}