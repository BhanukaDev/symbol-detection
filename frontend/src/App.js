import { useState, useRef, useEffect, useCallback } from "react";
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
  Card,
  CardMedia,
  Chip,
  Stack,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import ElectricalServicesIcon from "@mui/icons-material/ElectricalServices";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

const getMockupBOQ = () => {
  return [
    {
      id: 1,
      symbol: "Light",
      description: "Ceiling Light Fixture",
      quantity: Math.floor(Math.random() * 10) + 5,
      unit: "nos",
    },
    {
      id: 2,
      symbol: "Duplex Receptacle",
      description: "Double Outlet Socket",
      quantity: Math.floor(Math.random() * 15) + 8,
      unit: "nos",
    },
    {
      id: 3,
      symbol: "Single-pole, one-way switch",
      description: "Single Switch",
      quantity: Math.floor(Math.random() * 8) + 3,
      unit: "nos",
    },
    {
      id: 4,
      symbol: "Two-pole, one-way switch",
      description: "Double Switch",
      quantity: Math.floor(Math.random() * 5) + 2,
      unit: "nos",
    },
    {
      id: 5,
      symbol: "Three-pole, one-way switch",
      description: "Triple Switch",
      quantity: Math.floor(Math.random() * 3) + 1,
      unit: "nos",
    },
    {
      id: 6,
      symbol: "Two-way switch",
      description: "Two-way Switch",
      quantity: Math.floor(Math.random() * 4) + 2,
      unit: "nos",
    },
    {
      id: 7,
      symbol: "Junction Box",
      description: "Electrical Junction Box",
      quantity: Math.floor(Math.random() * 6) + 3,
      unit: "nos",
    },
  ];
};

const COLORS = {
  "Light": "#FF6B6B",
  "Duplex Receptacle": "#4ECDC4",
  "Single-pole, one-way switch": "#45B7D1",
  "Two-pole, one-way switch": "#96CEB4",
  "Three-pole, one-way switch": "#FFEAA7",
  "Two-way switch": "#DDA0DD",
  "Junction Box": "#FF8C42",
};

const API_URL = process.env.REACT_APP_API_URL || "https://sirbhanus-symbol-detection.hf.space";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [boqData, setBoqData] = useState(null);
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const drawDetections = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !detections.length) return;

    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const scaleX = rect.width / img.naturalWidth;
    const scaleY = rect.height / img.naturalHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const sx = x1 * scaleX;
      const sy = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;
      const color = COLORS[det.class_name] || "#00FF00";

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(sx, sy, sw, sh);

      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 12px Arial";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(sx, sy - 18, textWidth + 8, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(label, sx + 4, sy - 5);
    });
  }, [detections]);

  useEffect(() => {
    drawDetections();
    window.addEventListener("resize", drawDetections);
    return () => window.removeEventListener("resize", drawDetections);
  }, [drawDetections]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setBoqData(null);
      setDetections([]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);
      const response = await fetch(`${API_URL}/detect`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setBoqData(data.boq);
      setDetections(data.detections || []);
    } catch (err) {
      console.error("Detection failed:", err);
      // Fallback to mock data during development
      const mockData = getMockupBOQ();
      setBoqData(mockData);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setBoqData(null);
    setDetections([]);
  };

  const getTotalItems = () => {
    if (!boqData) return 0;
    return boqData.reduce((sum, item) => sum + item.quantity, 0);
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#f5f5f5", py: 4 }}>
      <Container maxWidth="md">
        {/* Header */}
        <Box sx={{ textAlign: "center", mb: 4 }}>
          <Stack
            direction="row"
            spacing={1}
            justifyContent="center"
            alignItems="center"
            sx={{ mb: 1 }}
          >
            <ElectricalServicesIcon
              sx={{ fontSize: 40, color: "primary.main" }}
            />
            <Typography variant="h4" component="h1" fontWeight="bold">
              Electrical Symbol Detector
            </Typography>
          </Stack>
          <Typography variant="subtitle1" color="text.secondary">
            Upload a building electrical plan to automatically generate Bill of
            Quantities (BOQ)
          </Typography>
        </Box>

        {/* Upload Section - Show when no BOQ data */}
        {!boqData && (
          <Paper sx={{ p: 4 }}>
            <Typography variant="h6" gutterBottom textAlign="center">
              Upload Electrical Plan
            </Typography>

            {/* Upload Area */}
            <Box
              component="label"
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                border: "2px dashed #1976d2",
                borderRadius: 2,
                p: 4,
                textAlign: "center",
                bgcolor: "#e3f2fd",
                mb: 3,
                cursor: "pointer",
                "&:hover": {
                  bgcolor: "#bbdefb",
                },
              }}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: "none" }}
              />
              <CloudUploadIcon
                sx={{ fontSize: 64, color: "primary.main", mb: 1 }}
              />
              <Typography variant="h6" color="primary">
                Click to upload or drag and drop
              </Typography>
              <Typography variant="body2" color="text.secondary">
                PNG, JPG up to 10MB
              </Typography>
            </Box>

            {/* Image Preview */}
            {imagePreview && (
              <Card sx={{ mb: 3 }}>
                <CardMedia
                  component="img"
                  image={imagePreview}
                  alt="Uploaded electrical plan"
                  sx={{
                    maxHeight: 400,
                    objectFit: "contain",
                    bgcolor: "#fff",
                  }}
                />
              </Card>
            )}

            {/* Analyze Button */}
            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={handleAnalyze}
              disabled={!selectedImage || loading}
              startIcon={
                loading ? <CircularProgress size={20} color="inherit" /> : null
              }
              sx={{ py: 1.5 }}
            >
              {loading ? "Analyzing..." : "Detect Symbols"}
            </Button>
          </Paper>
        )}

        {/* BOQ Results Section - Show when BOQ data exists */}
        {boqData && (
          <Paper sx={{ p: 4 }}>
            {/* Back Button */}
            <Button
              startIcon={<ArrowBackIcon />}
              onClick={handleReset}
              sx={{ mb: 3 }}
            >
              Upload New Image
            </Button>

            {/* Annotated Image */}
            {imagePreview && (
              <Box sx={{ position: "relative", mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Detected Symbols
                </Typography>
                <Box sx={{ position: "relative", display: "inline-block", width: "100%" }}>
                  <img
                    ref={imgRef}
                    src={imagePreview}
                    alt="Analyzed plan"
                    onLoad={drawDetections}
                    style={{
                      width: "100%",
                      display: "block",
                      borderRadius: 8,
                      border: "1px solid #ddd",
                    }}
                  />
                  <canvas
                    ref={canvasRef}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                </Box>
                {detections.length > 0 && (
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap", gap: 0.5 }}>
                    {Object.entries(COLORS).filter(([name]) =>
                      detections.some((d) => d.class_name === name)
                    ).map(([name, color]) => (
                      <Chip
                        key={name}
                        label={name}
                        size="small"
                        sx={{ bgcolor: color, color: "#000", fontWeight: "bold" }}
                      />
                    ))}
                  </Stack>
                )}
              </Box>
            )}

            {/* Header with total */}
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                mb: 3,
              }}
            >
              <Typography variant="h5" fontWeight="bold">
                Bill of Quantities (BOQ)
              </Typography>
              <Chip
                label={`Total: ${getTotalItems()} items`}
                color="primary"
                size="medium"
              />
            </Box>

            {/* BOQ Table */}
            <TableContainer sx={{ mb: 3 }}>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: "primary.main" }}>
                    <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                      #
                    </TableCell>
                    <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                      Symbol
                    </TableCell>
                    <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                      Description
                    </TableCell>
                    <TableCell
                      sx={{ color: "white", fontWeight: "bold" }}
                      align="center"
                    >
                      Qty
                    </TableCell>
                    <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                      Unit
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {boqData.map((row, index) => (
                    <TableRow
                      key={row.id}
                      sx={{
                        "&:nth-of-type(odd)": { bgcolor: "grey.50" },
                        "&:hover": { bgcolor: "grey.100" },
                      }}
                    >
                      <TableCell>{index + 1}</TableCell>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {row.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>{row.description}</TableCell>
                      <TableCell align="center">
                        <Chip
                          label={row.quantity}
                          size="small"
                          color="primary"
                        />
                      </TableCell>
                      <TableCell>{row.unit}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}
      </Container>
    </Box>
  );
}

export default App;
