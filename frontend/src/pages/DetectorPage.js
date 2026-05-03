import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Chip,
  Stack,
  Snackbar,
  Alert,
  Fade,
  IconButton,
  Tooltip,
  Grid,
  Slider,
  Collapse,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import SearchIcon from "@mui/icons-material/Search";
import ImageIcon from "@mui/icons-material/Image";
import DownloadIcon from "@mui/icons-material/Download";
import NavigateBeforeIcon from "@mui/icons-material/NavigateBefore";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import ZoomOutMapIcon from "@mui/icons-material/ZoomOutMap";
import TextFieldsIcon from "@mui/icons-material/TextFields";
import EditIcon from "@mui/icons-material/Edit";
import CheckIcon from "@mui/icons-material/Check";
import TuneIcon from "@mui/icons-material/Tune";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

const COLORS = {
  Light: "#FF6B6B",
  "Duplex Receptacle": "#4ECDC4",
  "Single-pole, one-way switch": "#45B7D1",
  "Two-pole, one-way switch": "#96CEB4",
  "Three-pole, one-way switch": "#FFEAA7",
  "Two-way switch": "#DDA0DD",
  "Junction Box": "#FF8C42",
};

const API_URL =
  process.env.REACT_APP_API_URL ||
  "https://sirbhanus-symbol-detection.hf.space";

const detSize = (d) => {
  const [x1, y1, x2, y2] = d.bbox;
  return Math.round(Math.sqrt((x2 - x1) * (y2 - y1)));
};

const iou = (a, b) => {
  const [ax1, ay1, ax2, ay2] = a.bbox;
  const [bx1, by1, bx2, by2] = b.bbox;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (!inter) return 0;
  return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter);
};

const nmsFilter = (dets, threshold) => {
  if (threshold >= 1) return dets;
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const det of sorted) {
    if (kept.every((k) => iou(k, det) < threshold)) kept.push(det);
  }
  return kept;
};

export default function DetectorPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [boqData, setBoqData] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [focusedIdx, setFocusedIdx] = useState(-1);
  const [showLabels, setShowLabels] = useState(false);
  const [sizeRange, setSizeRange] = useState([0, 9999]);
  const [sizeBounds, setSizeBounds] = useState([0, 9999]);
  const [confRange, setConfRange] = useState([0, 1]);
  const [confBounds, setConfBounds] = useState([0, 1]);
  const [overlapThresh, setOverlapThresh] = useState(1);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [hoveredIdx, setHoveredIdx] = useState(-1);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [classMenuPos, setClassMenuPos] = useState(null);
  const [classMenuDet, setClassMenuDet] = useState(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const transformRef = useRef(null);
  const wrapperRef = useRef(null);
  const mouseDownPos = useRef(null);

  // Detections passing the size filter (master list is never touched by slider)
  const visibleDetections = useMemo(() => {
    if (!detections.length) return [];
    const filtered = detections.filter((d) => {
      const s = detSize(d);
      return (
        s >= sizeRange[0] && s <= sizeRange[1] &&
        d.confidence >= confRange[0] && d.confidence <= confRange[1]
      );
    });
    return nmsFilter(filtered, overlapThresh);
  }, [detections, sizeRange, confRange, overlapThresh]);

  useEffect(() => {
    setFocusedIdx(-1);
  }, [sizeRange, confRange, overlapThresh]);

  const computedBoq = useMemo(() => {
    if (!visibleDetections.length) return [];
    const counts = {};
    visibleDetections.forEach((d) => {
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
    });
    return Object.entries(counts).map(([name, qty], i) => ({
      id: i + 1,
      symbol: name,
      description: name,
      quantity: qty,
      unit: "nos",
    }));
  }, [visibleDetections]);

  const drawDetections = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const scaleX = rect.width / img.naturalWidth;
    const scaleY = rect.height / img.naturalHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const isNavigating = focusedIdx >= 0;

    visibleDetections.forEach((det, idx) => {
      const isFocused = idx === focusedIdx;
      // In navigation mode, skip all non-focused detections entirely
      if (isNavigating && !isFocused) return;

      const [x1, y1, x2, y2] = det.bbox;
      const sx = x1 * scaleX;
      const sy = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;
      const color = COLORS[det.class_name] || "#00FF00";

      ctx.globalAlpha = 1;

      ctx.fillStyle = color + (isFocused ? "30" : "18");
      ctx.fillRect(sx, sy, sw, sh);

      ctx.strokeStyle = color;
      ctx.lineWidth = isFocused ? 3.5 : 2;
      ctx.strokeRect(sx, sy, sw, sh);

      if (isFocused) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.strokeRect(sx - 2, sy - 2, sw + 4, sh + 4);
      }

      // Show label when: navigating (always) OR showLabels toggle is on
      if (isNavigating || showLabels) {
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "600 11px Inter, Arial";
        const textWidth = ctx.measureText(label).width;
        const labelHeight = 20;
        const labelY = sy > labelHeight ? sy - labelHeight : sy;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(sx, labelY, textWidth + 10, labelHeight, 4);
        ctx.fill();
        ctx.fillStyle = "#000";
        ctx.fillText(label, sx + 5, labelY + 14);
      }

      ctx.globalAlpha = 1;
    });
  }, [visibleDetections, focusedIdx, showLabels]);

  useEffect(() => {
    drawDetections();
    window.addEventListener("resize", drawDetections);
    return () => window.removeEventListener("resize", drawDetections);
  }, [drawDetections]);

  useEffect(() => {
    if (focusedIdx < 0 || !visibleDetections[focusedIdx] || !imgRef.current || !transformRef.current || !wrapperRef.current) return;

    requestAnimationFrame(() => {
      const img = imgRef.current;
      const api = transformRef.current;
      const wrapper = api?.instance?.wrapperComponent || wrapperRef.current;
      const content = api?.instance?.contentComponent || img;

      if (!img || !wrapper) return;

      const ww = wrapper.clientWidth;
      const wh = wrapper.clientHeight;
      const imgW = content.clientWidth || img.clientWidth;
      const imgH = content.clientHeight || img.clientHeight;

      const det = visibleDetections[focusedIdx];
      const [x1, y1, x2, y2] = det.bbox;

      const sx = imgW / img.naturalWidth;
      const sy = imgH / img.naturalHeight;

      const cx = ((x1 + x2) / 2) * sx;
      const cy = ((y1 + y2) / 2) * sy;
      const dw = (x2 - x1) * sx;
      const dh = (y2 - y1) * sy;

      const zoom = Math.min(ww / (dw * 3), wh / (dh * 3), 5);
      const offsetX = ww / 2 - cx * zoom;
      const offsetY = wh / 2 - cy * zoom;

      api.setTransform(offsetX, offsetY, zoom, 300, "easeOut");
    });
  }, [focusedIdx, visibleDetections]);

  const hitTest = useCallback((px, py, rect) => {
    const img = imgRef.current;
    if (!img || !rect) return -1;
    const scaleX = rect.width / img.naturalWidth;
    const scaleY = rect.height / img.naturalHeight;
    const indices =
      focusedIdx >= 0
        ? [focusedIdx]
        : [...Array(visibleDetections.length).keys()].reverse();
    for (const i of indices) {
      if (i >= visibleDetections.length) continue;
      const [x1, y1, x2, y2] = visibleDetections[i].bbox;
      if (px >= x1 * scaleX && px <= x2 * scaleX && py >= y1 * scaleY && py <= y2 * scaleY)
        return i;
    }
    return -1;
  }, [visibleDetections, focusedIdx]);

  const handleCanvasMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const idx = hitTest(px, py, rect);
    setHoveredIdx(idx);
    if (idx >= 0) setHoverPos({ x: px, y: py });
    e.currentTarget.style.cursor = idx >= 0 ? "pointer" : "default";
  }, [hitTest]);

  const handleCanvasMouseLeave = useCallback(() => setHoveredIdx(-1), []);

  const handleCanvasMouseDown = useCallback((e) => {
    mouseDownPos.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleCanvasClick = useCallback((e) => {
    if (mouseDownPos.current) {
      const dx = e.clientX - mouseDownPos.current.x;
      const dy = e.clientY - mouseDownPos.current.y;
      if (Math.sqrt(dx * dx + dy * dy) > 5) return; // was a drag
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const idx = hitTest(px, py, rect);
    if (idx >= 0) {
      setFocusedIdx(idx);
      setClassMenuPos({ x: e.clientX, y: e.clientY });
      setClassMenuDet(visibleDetections[idx]);
    }
  }, [hitTest, visibleDetections]);

  const handleClassChange = (newClass) => {
    if (!classMenuDet) return;
    setDetections((prev) =>
      prev.map((d) => (d === classMenuDet ? { ...d, class_name: newClass } : d))
    );
    setClassMenuPos(null);
    setClassMenuDet(null);
  };

  const handleFile = (file) => {
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setBoqData(null);
      setDetections([]);
      setFocusedIdx(-1);
    }
  };

  const handleImageUpload = (event) => handleFile(event.target.files[0]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    setLoading(true);
    setFocusedIdx(-1);
    setShowLabels(false);
    try {
      const formData = new FormData();
      formData.append("image", selectedImage);
      const response = await fetch(`${API_URL}/detect`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setBoqData(data.boq);
      const dets = data.detections || [];
      setDetections(dets);
      if (dets.length > 0) {
        const sizes = dets.map(detSize);
        setSizeBounds([Math.min(...sizes), Math.max(...sizes)]);
        setSizeRange([Math.min(...sizes), Math.max(...sizes)]);
        const confs = dets.map((d) => d.confidence);
        setConfBounds([Math.min(...confs), Math.max(...confs)]);
        setConfRange([Math.min(...confs), Math.max(...confs)]);
        setOverlapThresh(1);
      }
    } catch (err) {
      console.error("Detection failed:", err);
      setError(`Detection failed: ${err.message}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setBoqData(null);
    setDetections([]);
    setFocusedIdx(-1);
    setClassMenuPos(null);
    setClassMenuDet(null);
    setSizeRange([0, 9999]);
    setSizeBounds([0, 9999]);
    setConfRange([0, 1]);
    setConfBounds([0, 1]);
    setOverlapThresh(1);
  };

  const handleRemoveDetection = () => {
    if (focusedIdx < 0 || focusedIdx >= visibleDetections.length) return;
    const toRemove = visibleDetections[focusedIdx];
    const next = detections.filter((d) => d !== toRemove);
    setDetections(next);
    if (visibleDetections.length - 1 === 0) {
      setFocusedIdx(-1);
    } else if (focusedIdx >= visibleDetections.length - 1) {
      setFocusedIdx(visibleDetections.length - 2);
    }
  };

  const handleResetView = () => {
    setFocusedIdx(-1);
    transformRef.current?.resetTransform(300, "easeOut");
  };

  const handlePrev = () => {
    if (!visibleDetections.length) return;
    setFocusedIdx((prev) =>
      prev <= 0 ? visibleDetections.length - 1 : prev - 1
    );
  };

  const handleNext = () => {
    if (!visibleDetections.length) return;
    setFocusedIdx((prev) =>
      prev >= visibleDetections.length - 1 ? 0 : prev + 1
    );
  };

  const displayBoq = computedBoq;
  const getTotalItems = () =>
    displayBoq.reduce((sum, item) => sum + item.quantity, 0);

  const handleDownloadCSV = () => {
    if (!displayBoq.length) return;
    const header = "No,Symbol,Description,Quantity,Unit\n";
    const rows = displayBoq
      .map(
        (row, i) =>
          `${i + 1},"${row.symbol}","${row.description}",${row.quantity},"${row.unit}"`
      )
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "boq_report.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const focusedDet = focusedIdx >= 0 ? visibleDetections[focusedIdx] : null;
  const showSizeFilter = sizeBounds[0] < sizeBounds[1];
  const showConfFilter = confBounds[0] < confBounds[1];
  const showOverlapFilter = detections.length > 1;
  const showFilters = showSizeFilter || showConfFilter || showOverlapFilter;
  const hiddenCount = detections.length - visibleDetections.length;

  return (
    <Box sx={{ py: 5, minHeight: "calc(100vh - 64px)" }}>
      <Container maxWidth={boqData ? "xl" : "md"}>
        {!boqData && (
          <Fade in timeout={600}>
            <Box sx={{ textAlign: "center", mb: 5 }}>
              <Typography
                variant="h3"
                gutterBottom
                sx={{
                  background:
                    "linear-gradient(135deg, #1a237e 0%, #00bfa5 100%)",
                  backgroundClip: "text",
                  WebkitBackgroundClip: "text",
                  color: "transparent",
                }}
              >
                Detect Symbols
              </Typography>
              <Typography
                variant="h6"
                color="text.secondary"
                fontWeight={400}
              >
                Upload an electrical floor plan and get an instant Bill of
                Quantities
              </Typography>
            </Box>
          </Fade>
        )}

        {!boqData && (
          <Fade in timeout={800}>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 3, md: 5 },
                border: "1px solid",
                borderColor: "divider",
              }}
            >
              <Box
                component="label"
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "2px dashed",
                  borderColor: dragActive ? "secondary.main" : "grey.300",
                  borderRadius: 3,
                  p: { xs: 4, md: 6 },
                  textAlign: "center",
                  bgcolor: dragActive ? "rgba(0,191,165,0.03)" : "grey.50",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  "&:hover": {
                    borderColor: "secondary.main",
                    bgcolor: "rgba(0,191,165,0.03)",
                  },
                }}
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={{ display: "none" }}
                />
                {imagePreview ? (
                  <Box sx={{ width: "100%" }}>
                    <img
                      src={imagePreview}
                      alt="Uploaded plan"
                      style={{
                        maxWidth: "100%",
                        maxHeight: 400,
                        objectFit: "contain",
                        borderRadius: 8,
                      }}
                    />
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mt: 1 }}
                    >
                      {selectedImage?.name} — click to change
                    </Typography>
                  </Box>
                ) : (
                  <>
                    <ImageIcon
                      sx={{ fontSize: 56, color: "grey.400", mb: 2 }}
                    />
                    <Typography variant="h6" color="text.secondary">
                      Drop your floor plan here
                    </Typography>
                    <Typography variant="body2" color="text.disabled">
                      or click to browse — PNG, JPG up to 10MB
                    </Typography>
                  </>
                )}
              </Box>

              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={handleAnalyze}
                disabled={!selectedImage || loading}
                startIcon={
                  loading ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    <SearchIcon />
                  )
                }
                sx={{
                  mt: 3,
                  py: 1.5,
                  bgcolor: "secondary.main",
                  "&:hover": { bgcolor: "secondary.dark" },
                }}
              >
                {loading ? "Analyzing…" : "Detect Symbols"}
              </Button>
            </Paper>
          </Fade>
        )}

        {/* ===== RESULTS ===== */}
        {boqData && (
          <Fade in timeout={500}>
            <Box>
              <Stack
                direction="row"
                justifyContent="space-between"
                alignItems="center"
                sx={{ mb: 2 }}
              >
                <Button
                  startIcon={<ArrowBackIcon />}
                  onClick={handleReset}
                  sx={{ color: "text.secondary" }}
                >
                  New Image
                </Button>
                <Stack direction="row" spacing={1}>
                  <Chip
                    label={`${visibleDetections.length} detections`}
                    size="small"
                    sx={{ bgcolor: "secondary.main", color: "white" }}
                  />
                  {hiddenCount > 0 && (
                    <Chip
                      label={`${hiddenCount} filtered`}
                      size="small"
                      variant="outlined"
                      sx={{ color: "text.secondary" }}
                    />
                  )}
                  <Chip
                    label={`${getTotalItems()} items`}
                    size="small"
                    color="primary"
                  />
                </Stack>
              </Stack>

              <Grid container spacing={2.5}>
                <Grid size={{ xs: 12, lg: 7 }}>
                  <Paper
                    elevation={0}
                    sx={{
                      border: "1px solid",
                      borderColor: "divider",
                      overflow: "hidden",
                    }}
                  >
                    {/* toolbar */}
                    <Stack
                      direction="row"
                      alignItems="center"
                      justifyContent="space-between"
                      sx={{
                        px: 2,
                        py: 1,
                        bgcolor: "grey.50",
                        borderBottom: "1px solid",
                        borderColor: "divider",
                      }}
                    >
                      <Typography variant="subtitle2" color="text.secondary">
                        {focusedIdx >= 0
                          ? `Detection ${focusedIdx + 1} / ${visibleDetections.length}`
                          : "Overview — scroll to zoom"}
                      </Typography>
                      <Stack direction="row" spacing={0.5} alignItems="center">
                        <Tooltip title={showLabels ? "Hide labels" : "Show labels"}>
                          <IconButton
                            size="small"
                            onClick={() => setShowLabels((v) => !v)}
                            sx={{ color: showLabels ? "secondary.main" : "text.secondary" }}
                          >
                            <TextFieldsIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Previous (←)">
                          <IconButton size="small" onClick={handlePrev} disabled={!visibleDetections.length}>
                            <NavigateBeforeIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Next (→)">
                          <IconButton size="small" onClick={handleNext} disabled={!visibleDetections.length}>
                            <NavigateNextIcon />
                          </IconButton>
                        </Tooltip>
                        {focusedIdx >= 0 && (
                          <>
                            <Tooltip title="Remove this detection">
                              <IconButton
                                size="small"
                                onClick={handleRemoveDetection}
                                sx={{ color: "error.main" }}
                              >
                                <DeleteOutlineIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Back to overview">
                              <IconButton size="small" onClick={handleResetView}>
                                <ZoomOutMapIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </>
                        )}
                      </Stack>
                    </Stack>

                    {showFilters && (
                      <>
                        <Box
                          onClick={() => setFiltersOpen((v) => !v)}
                          sx={{
                            px: 2.5,
                            py: 0.75,
                            borderBottom: "1px solid",
                            borderColor: "divider",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            cursor: "pointer",
                            userSelect: "none",
                            "&:hover": { bgcolor: "grey.50" },
                          }}
                        >
                          <Stack direction="row" spacing={0.75} alignItems="center">
                            <TuneIcon sx={{ fontSize: 14, color: "text.secondary" }} />
                            <Typography variant="caption" color="text.secondary" fontWeight={600}>
                              Detection Filters
                            </Typography>
                            {hiddenCount > 0 && (
                              <Typography variant="caption" color="secondary.main" fontWeight={600}>
                                · {hiddenCount} hidden
                              </Typography>
                            )}
                          </Stack>
                          <ExpandMoreIcon
                            sx={{
                              fontSize: 16,
                              color: "text.secondary",
                              transform: filtersOpen ? "rotate(180deg)" : "rotate(0deg)",
                              transition: "transform 0.2s",
                            }}
                          />
                        </Box>
                        <Collapse in={filtersOpen}>
                          <Box
                            sx={{
                              px: 2.5,
                              pt: 1.5,
                              pb: 1,
                              borderBottom: "1px solid",
                              borderColor: "divider",
                            }}
                          >
                            {showSizeFilter && (
                              <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: showConfFilter || showOverlapFilter ? 1 : 0 }}>
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 72 }}>
                                  Size
                                </Typography>
                                <Slider
                                  value={sizeRange}
                                  onChange={(_, v) => setSizeRange(v)}
                                  min={sizeBounds[0]}
                                  max={sizeBounds[1]}
                                  valueLabelDisplay="auto"
                                  valueLabelFormat={(v) => `${v}px`}
                                  size="small"
                                  disableSwap
                                  sx={{ color: "secondary.main" }}
                                />
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 88, textAlign: "right" }}>
                                  {sizeRange[0]}–{sizeRange[1]}px
                                </Typography>
                              </Stack>
                            )}
                            {showConfFilter && (
                              <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: showOverlapFilter ? 1 : 0 }}>
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 72 }}>
                                  Confidence
                                </Typography>
                                <Slider
                                  value={confRange}
                                  onChange={(_, v) => setConfRange(v)}
                                  min={confBounds[0]}
                                  max={confBounds[1]}
                                  step={0.01}
                                  valueLabelDisplay="auto"
                                  valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
                                  size="small"
                                  disableSwap
                                  sx={{ color: "secondary.main" }}
                                />
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 88, textAlign: "right" }}>
                                  {(confRange[0] * 100).toFixed(0)}–{(confRange[1] * 100).toFixed(0)}%
                                </Typography>
                              </Stack>
                            )}
                            {showOverlapFilter && (
                              <Stack direction="row" spacing={2} alignItems="center">
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 72 }}>
                                  Max overlap
                                </Typography>
                                <Slider
                                  value={overlapThresh}
                                  onChange={(_, v) => setOverlapThresh(v)}
                                  min={0}
                                  max={1}
                                  step={0.05}
                                  valueLabelDisplay="auto"
                                  valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
                                  size="small"
                                  sx={{ color: "secondary.main" }}
                                />
                                <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 88, textAlign: "right" }}>
                                  {(overlapThresh * 100).toFixed(0)}% IoU
                                </Typography>
                              </Stack>
                            )}
                          </Box>
                        </Collapse>
                      </>
                    )}

                    {focusedDet && (
                      <Box sx={{ px: 2, pt: 1 }}>
                        <Chip
                          icon={<CenterFocusStrongIcon />}
                          label={`${focusedDet.class_name} — ${(focusedDet.confidence * 100).toFixed(0)}% confidence`}
                          sx={{
                            bgcolor: (COLORS[focusedDet.class_name] || "#ccc") + "25",
                            border: "1px solid " + (COLORS[focusedDet.class_name] || "#ccc"),
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                    )}

                    {imagePreview && (
                      <Box sx={{ p: 1 }}>
                        <TransformWrapper
                          ref={transformRef}
                          initialScale={1}
                          minScale={0.5}
                          maxScale={8}
                          centerOnInit
                          wheel={{ step: 0.08 }}
                          doubleClick={{ mode: "reset" }}
                        >
                          <TransformComponent
                            wrapperStyle={{
                              width: "100%",
                              maxHeight: "65vh",
                              borderRadius: 8,
                              overflow: "hidden",
                              background: "#f5f5f5",
                            }}
                            wrapperProps={{ ref: wrapperRef }}
                            contentStyle={{ width: "100%" }}
                          >
                            <Box
                              sx={{
                                position: "relative",
                                display: "inline-block",
                                width: "100%",
                              }}
                            >
                              <img
                                ref={imgRef}
                                src={imagePreview}
                                alt="Analyzed plan"
                                onLoad={drawDetections}
                                style={{ width: "100%", display: "block" }}
                              />
                              <canvas
                                ref={canvasRef}
                                onMouseMove={handleCanvasMouseMove}
                                onMouseLeave={handleCanvasMouseLeave}
                                onMouseDown={handleCanvasMouseDown}
                                onClick={handleCanvasClick}
                                style={{
                                  position: "absolute",
                                  top: 0,
                                  left: 0,
                                  width: "100%",
                                  height: "100%",
                                }}
                              />
                              {hoveredIdx >= 0 && visibleDetections[hoveredIdx] && (
                                <Box
                                  sx={{
                                    position: "absolute",
                                    left: hoverPos.x + 12,
                                    top: hoverPos.y - 28,
                                    bgcolor: "rgba(0,0,0,0.78)",
                                    color: "#fff",
                                    px: 1,
                                    py: 0.25,
                                    borderRadius: 1,
                                    fontSize: "0.72rem",
                                    fontWeight: 600,
                                    pointerEvents: "none",
                                    whiteSpace: "nowrap",
                                    zIndex: 10,
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 0.5,
                                  }}
                                >
                                  <Box
                                    sx={{
                                      width: 8,
                                      height: 8,
                                      borderRadius: "50%",
                                      bgcolor: COLORS[visibleDetections[hoveredIdx].class_name] || "#ccc",
                                      flexShrink: 0,
                                    }}
                                  />
                                  {visibleDetections[hoveredIdx].class_name}{" "}
                                  {(visibleDetections[hoveredIdx].confidence * 100).toFixed(0)}%
                                  <EditIcon sx={{ fontSize: 10, opacity: 0.7, ml: 0.25 }} />
                                </Box>
                              )}
                            </Box>
                          </TransformComponent>
                        </TransformWrapper>
                      </Box>
                    )}

                    <Stack
                      direction="row"
                      spacing={0.5}
                      sx={{ px: 2, pb: 1.5, flexWrap: "wrap", gap: 0.5 }}
                    >
                      {Object.entries(COLORS)
                        .filter(([name]) =>
                          visibleDetections.some((d) => d.class_name === name)
                        )
                        .map(([name, color]) => (
                          <Chip
                            key={name}
                            label={`${name} (${visibleDetections.filter((d) => d.class_name === name).length})`}
                            size="small"
                            sx={{
                              bgcolor: color + "25",
                              border: "1px solid " + color,
                              fontWeight: 600,
                              fontSize: "0.7rem",
                            }}
                          />
                        ))}
                    </Stack>
                  </Paper>
                </Grid>

                <Grid size={{ xs: 12, lg: 5 }}>
                  <Paper
                    elevation={0}
                    sx={{
                      border: "1px solid",
                      borderColor: "divider",
                      overflow: "hidden",
                      position: { lg: "sticky" },
                      top: { lg: 80 },
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        px: 2,
                        py: 1.5,
                        bgcolor: "primary.main",
                        color: "white",
                      }}
                    >
                      <Typography variant="subtitle1" fontWeight={700}>
                        Bill of Quantities
                      </Typography>
                      <Tooltip title="Download CSV">
                        <IconButton
                          size="small"
                          onClick={handleDownloadCSV}
                          sx={{ color: "white" }}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    <TableContainer sx={{ maxHeight: { lg: "60vh" } }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ fontWeight: 700, width: 40 }}>
                              #
                            </TableCell>
                            <TableCell sx={{ fontWeight: 700 }}>
                              Symbol
                            </TableCell>
                            <TableCell sx={{ fontWeight: 700 }} align="center">
                              Qty
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {displayBoq.map((row, index) => (
                            <TableRow
                              key={row.id}
                              sx={{
                                "&:hover": { bgcolor: "grey.50" },
                                "&:last-child td": { borderBottom: 0 },
                              }}
                            >
                              <TableCell>{index + 1}</TableCell>
                              <TableCell>
                                <Stack
                                  direction="row"
                                  spacing={1}
                                  alignItems="center"
                                >
                                  <Box
                                    sx={{
                                      width: 10,
                                      height: 10,
                                      borderRadius: "50%",
                                      bgcolor: COLORS[row.symbol] || "grey.400",
                                      flexShrink: 0,
                                    }}
                                  />
                                  <Typography variant="body2" fontWeight={600}>
                                    {row.symbol}
                                  </Typography>
                                </Stack>
                              </TableCell>
                              <TableCell align="center">
                                <Chip
                                  label={row.quantity}
                                  size="small"
                                  sx={{
                                    bgcolor: "secondary.main",
                                    color: "white",
                                    fontWeight: 700,
                                    minWidth: 28,
                                    height: 24,
                                  }}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        px: 2,
                        py: 1.5,
                        bgcolor: "grey.50",
                        borderTop: "1px solid",
                        borderColor: "divider",
                      }}
                    >
                      <Typography variant="body2" fontWeight={700}>
                        Total
                      </Typography>
                      <Typography variant="body2" fontWeight={700}>
                        {getTotalItems()} items
                      </Typography>
                    </Box>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          </Fade>
        )}
      </Container>

      <Menu
        open={!!classMenuPos}
        onClose={() => { setClassMenuPos(null); setClassMenuDet(null); }}
        anchorReference="anchorPosition"
        anchorPosition={classMenuPos ? { top: classMenuPos.y, left: classMenuPos.x } : undefined}
      >
        <Typography variant="caption" color="text.secondary" sx={{ px: 2, py: 0.5, display: "block" }}>
          Change class
        </Typography>
        {Object.entries(COLORS).map(([cls, color]) => (
          <MenuItem key={cls} onClick={() => handleClassChange(cls)} dense>
            <ListItemIcon sx={{ minWidth: 28 }}>
              <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: color }} />
            </ListItemIcon>
            <ListItemText primaryTypographyProps={{ variant: "body2" }}>
              {cls}
            </ListItemText>
            {classMenuDet?.class_name === cls && (
              <CheckIcon sx={{ fontSize: 14, color: "secondary.main", ml: 1 }} />
            )}
          </MenuItem>
        ))}
      </Menu>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setError(null)}
          severity="error"
          variant="filled"
        >
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
}
