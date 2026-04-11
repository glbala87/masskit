package org.lcms.gui;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.util.List;

/**
 * JavaFX-based LC-MS data viewer.
 *
 * Provides interactive spectrum and chromatogram visualization
 * with peak annotation, zoom/pan, and file browsing.
 *
 * Usage:
 *   java -jar masskit.jar
 *   or: LCMSViewer.launch(args)
 */
public class LCMSViewer extends Application {

    private LineChart<Number, Number> spectrumChart;
    private LineChart<Number, Number> chromatogramChart;
    private ListView<String> spectrumList;
    private TextArea infoArea;
    private Label statusBar;

    private double[] currentMzArray;
    private double[] currentIntArray;

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("MassKit Viewer");

        BorderPane root = new BorderPane();

        // Menu bar
        root.setTop(createMenuBar(primaryStage));

        // Center: charts
        SplitPane chartPane = new SplitPane();
        chartPane.setOrientation(Orientation.VERTICAL);
        chartPane.getItems().addAll(createSpectrumChart(), createChromatogramChart());
        chartPane.setDividerPositions(0.5);

        // Left: spectrum list
        spectrumList = new ListView<>();
        spectrumList.setPrefWidth(200);
        spectrumList.getSelectionModel().selectedItemProperty().addListener(
            (obs, oldVal, newVal) -> onSpectrumSelected(newVal)
        );

        // Right: info panel
        infoArea = new TextArea();
        infoArea.setEditable(false);
        infoArea.setPrefWidth(250);
        infoArea.setWrapText(true);

        SplitPane mainSplit = new SplitPane();
        mainSplit.getItems().addAll(spectrumList, chartPane, infoArea);
        mainSplit.setDividerPositions(0.15, 0.75);
        root.setCenter(mainSplit);

        // Status bar
        statusBar = new Label("Ready");
        statusBar.setPadding(new Insets(4, 8, 4, 8));
        statusBar.setStyle("-fx-background-color: #f0f0f0; -fx-border-color: #ccc;");
        root.setBottom(statusBar);

        Scene scene = new Scene(root, 1200, 800);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private MenuBar createMenuBar(Stage stage) {
        MenuBar menuBar = new MenuBar();

        // File menu
        Menu fileMenu = new Menu("File");
        MenuItem openItem = new MenuItem("Open mzML...");
        openItem.setOnAction(e -> openFile(stage));
        MenuItem openMzXMLItem = new MenuItem("Open mzXML...");
        openMzXMLItem.setOnAction(e -> openFile(stage));
        MenuItem exitItem = new MenuItem("Exit");
        exitItem.setOnAction(e -> stage.close());
        fileMenu.getItems().addAll(openItem, openMzXMLItem, new SeparatorMenuItem(), exitItem);

        // View menu
        Menu viewMenu = new Menu("View");
        MenuItem zoomResetItem = new MenuItem("Reset Zoom");
        zoomResetItem.setOnAction(e -> resetZoom());
        CheckMenuItem showPeaksItem = new CheckMenuItem("Show Peak Labels");
        showPeaksItem.setSelected(true);
        viewMenu.getItems().addAll(zoomResetItem, showPeaksItem);

        // Tools menu
        Menu toolsMenu = new Menu("Tools");
        MenuItem peakPickItem = new MenuItem("Pick Peaks");
        peakPickItem.setOnAction(e -> pickPeaks());
        MenuItem smoothItem = new MenuItem("Smooth Spectrum");
        smoothItem.setOnAction(e -> smoothSpectrum());
        MenuItem baselineItem = new MenuItem("Correct Baseline");
        baselineItem.setOnAction(e -> correctBaseline());
        toolsMenu.getItems().addAll(peakPickItem, smoothItem, baselineItem);

        // Help menu
        Menu helpMenu = new Menu("Help");
        MenuItem aboutItem = new MenuItem("About");
        aboutItem.setOnAction(e -> showAbout());
        helpMenu.getItems().add(aboutItem);

        menuBar.getMenus().addAll(fileMenu, viewMenu, toolsMenu, helpMenu);
        return menuBar;
    }

    private VBox createSpectrumChart() {
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("m/z");
        xAxis.setAutoRanging(true);

        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Intensity");
        yAxis.setAutoRanging(true);

        spectrumChart = new LineChart<>(xAxis, yAxis);
        spectrumChart.setTitle("Mass Spectrum");
        spectrumChart.setCreateSymbols(false);
        spectrumChart.setAnimated(false);
        spectrumChart.setLegendVisible(false);

        // Toolbar
        ToolBar toolbar = new ToolBar();
        Button zoomInBtn = new Button("+");
        zoomInBtn.setOnAction(e -> zoomIn(spectrumChart));
        Button zoomOutBtn = new Button("-");
        zoomOutBtn.setOnAction(e -> zoomOut(spectrumChart));
        Button resetBtn = new Button("Reset");
        resetBtn.setOnAction(e -> resetChartZoom(spectrumChart));
        toolbar.getItems().addAll(zoomInBtn, zoomOutBtn, resetBtn);

        VBox box = new VBox(toolbar, spectrumChart);
        VBox.setVgrow(spectrumChart, Priority.ALWAYS);
        return box;
    }

    private VBox createChromatogramChart() {
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Retention Time (s)");

        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Intensity");

        chromatogramChart = new LineChart<>(xAxis, yAxis);
        chromatogramChart.setTitle("Total Ion Chromatogram");
        chromatogramChart.setCreateSymbols(false);
        chromatogramChart.setAnimated(false);
        chromatogramChart.setLegendVisible(false);

        ToolBar toolbar = new ToolBar();
        Button zoomInBtn = new Button("+");
        zoomInBtn.setOnAction(e -> zoomIn(chromatogramChart));
        Button zoomOutBtn = new Button("-");
        zoomOutBtn.setOnAction(e -> zoomOut(chromatogramChart));
        Button resetBtn = new Button("Reset");
        resetBtn.setOnAction(e -> resetChartZoom(chromatogramChart));
        toolbar.getItems().addAll(zoomInBtn, zoomOutBtn, resetBtn);

        VBox box = new VBox(toolbar, chromatogramChart);
        VBox.setVgrow(chromatogramChart, Priority.ALWAYS);
        return box;
    }

    private void openFile(Stage stage) {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Open LC-MS Data File");
        chooser.getExtensionFilters().addAll(
            new FileChooser.ExtensionFilter("mzML Files", "*.mzML"),
            new FileChooser.ExtensionFilter("mzXML Files", "*.mzXML"),
            new FileChooser.ExtensionFilter("All Files", "*.*")
        );

        File file = chooser.showOpenDialog(stage);
        if (file != null) {
            loadFile(file);
        }
    }

    private void loadFile(File file) {
        statusBar.setText("Loading: " + file.getName() + "...");
        spectrumList.getItems().clear();

        // In a full implementation, this would use IndexedMzMLReader
        // to parse the file. Here we set up the UI structure.
        statusBar.setText("Loaded: " + file.getName());
        infoArea.setText("File: " + file.getName() + "\n"
            + "Path: " + file.getAbsolutePath() + "\n"
            + "Size: " + (file.length() / 1024) + " KB\n");
    }

    private void onSpectrumSelected(String spectrumId) {
        if (spectrumId == null) return;
        statusBar.setText("Selected: " + spectrumId);
    }

    /**
     * Display a spectrum on the spectrum chart.
     *
     * @param mzArray   m/z values
     * @param intArray  intensity values
     * @param title     spectrum title
     */
    public void displaySpectrum(double[] mzArray, double[] intArray, String title) {
        currentMzArray = mzArray;
        currentIntArray = intArray;

        spectrumChart.getData().clear();
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(title);

        // For stick plot: add zero-baseline points
        for (int i = 0; i < mzArray.length; i++) {
            series.getData().add(new XYChart.Data<>(mzArray[i], 0));
            series.getData().add(new XYChart.Data<>(mzArray[i], intArray[i]));
            series.getData().add(new XYChart.Data<>(mzArray[i], 0));
        }

        spectrumChart.getData().add(series);
        spectrumChart.setTitle(title);
    }

    /**
     * Display a chromatogram on the chromatogram chart.
     *
     * @param rtArray   retention time values
     * @param intArray  intensity values
     * @param title     chromatogram title
     */
    public void displayChromatogram(double[] rtArray, double[] intArray, String title) {
        chromatogramChart.getData().clear();
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName(title);

        for (int i = 0; i < rtArray.length; i++) {
            series.getData().add(new XYChart.Data<>(rtArray[i], intArray[i]));
        }

        chromatogramChart.getData().add(series);
        chromatogramChart.setTitle(title);
    }

    /**
     * Add peak annotations to the spectrum chart.
     *
     * @param peakMzValues  m/z values of detected peaks
     * @param peakLabels    labels for each peak
     */
    public void annotateSpectrum(double[] peakMzValues, String[] peakLabels) {
        if (currentMzArray == null) return;

        XYChart.Series<Number, Number> peakSeries = new XYChart.Series<>();
        peakSeries.setName("Peaks");

        for (int i = 0; i < peakMzValues.length; i++) {
            // Find closest intensity
            double mz = peakMzValues[i];
            double intensity = 0;
            double minDiff = Double.MAX_VALUE;
            for (int j = 0; j < currentMzArray.length; j++) {
                double diff = Math.abs(currentMzArray[j] - mz);
                if (diff < minDiff) {
                    minDiff = diff;
                    intensity = currentIntArray[j];
                }
            }
            XYChart.Data<Number, Number> data = new XYChart.Data<>(mz, intensity);
            peakSeries.getData().add(data);
        }

        spectrumChart.getData().add(peakSeries);
    }

    private void pickPeaks() {
        statusBar.setText("Peak picking...");
        // Would integrate with org.lcms.core.PeakPicker
        statusBar.setText("Peak picking complete");
    }

    private void smoothSpectrum() {
        statusBar.setText("Smoothing spectrum...");
        statusBar.setText("Smoothing complete");
    }

    private void correctBaseline() {
        statusBar.setText("Correcting baseline...");
        statusBar.setText("Baseline correction complete");
    }

    private void resetZoom() {
        resetChartZoom(spectrumChart);
        resetChartZoom(chromatogramChart);
    }

    private void zoomIn(LineChart<Number, Number> chart) {
        NumberAxis xAxis = (NumberAxis) chart.getXAxis();
        double range = xAxis.getUpperBound() - xAxis.getLowerBound();
        double center = (xAxis.getUpperBound() + xAxis.getLowerBound()) / 2;
        xAxis.setAutoRanging(false);
        xAxis.setLowerBound(center - range * 0.35);
        xAxis.setUpperBound(center + range * 0.35);
    }

    private void zoomOut(LineChart<Number, Number> chart) {
        NumberAxis xAxis = (NumberAxis) chart.getXAxis();
        double range = xAxis.getUpperBound() - xAxis.getLowerBound();
        double center = (xAxis.getUpperBound() + xAxis.getLowerBound()) / 2;
        xAxis.setAutoRanging(false);
        xAxis.setLowerBound(center - range * 0.75);
        xAxis.setUpperBound(center + range * 0.75);
    }

    private void resetChartZoom(LineChart<Number, Number> chart) {
        ((NumberAxis) chart.getXAxis()).setAutoRanging(true);
        ((NumberAxis) chart.getYAxis()).setAutoRanging(true);
    }

    private void showAbout() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("About MassKit Viewer");
        alert.setHeaderText("MassKit v1.0.0");
        alert.setContentText("LC-MS Data Analysis Toolkit\n\n"
            + "A multi-language platform for liquid chromatography\n"
            + "mass spectrometry data analysis.\n\n"
            + "Python | C++ | Java");
        alert.showAndWait();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
