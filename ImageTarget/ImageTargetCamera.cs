using OpenCvSharp;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;

public class ImageTargetCamera : MonoBehaviour
{
    public int deviceId = 0;
    [HideInInspector] public Camera camera;
    [HideInInspector] public RawImage rawImage;
    public GameObject marker_obj;
    public bool shouldDrawRect = false;
    public bool shouldDrawAxis = false;

    private WebCamTexture webcamTexture;
    private MarkerBehavior marker;
    private Mat cameraMatrix;
    private Mat distortionMatrix;
    private ORB detector;
    private DescriptorMatcher matcher;

    private Mat previous_frameMatGray;
    private List<Point2f> previous_points2dFrame;

    // Start is called before the first frame update
    void Start()
    {
        WebCamDevice[] webCamDevices = WebCamTexture.devices;
        Assert.AreNotEqual(0, webCamDevices.Length);
        webcamTexture = new WebCamTexture(webCamDevices[deviceId].name, 640, 480);
        webcamTexture.Play();

        //------------------------
        float width = 640;
        float height = 480;

        float imageSizeScale = 1.0f;
        float widthScale = (float)Screen.width / width;
        float heightScale = (float)Screen.height / height;
        if (widthScale < heightScale)
        {
            camera.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            imageSizeScale = (float)Screen.height / (float)Screen.width;
        }
        else
        {
            camera.orthographicSize = height / 2;
        }

        //set cameraparam
        int max_d = (int)Mathf.Max(width, height);
        float fx = max_d;
        float fy = max_d;
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        cameraMatrix = new Mat(3, 3, MatType.CV_64FC1, new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 });
        distortionMatrix = new Mat(5, 1, MatType.CV_64FC1, new double[] { 0, 0, 0, 0, 0 });

        //calibration camera
        Size imageSize = new Size(width * imageSizeScale, height * imageSizeScale);
        double apertureWidth = 0;
        double apertureHeight = 0;
        double fovx;
        double fovy;
        double focalLength;
        Point2d principalPoint;
        double aspectratio;

        Cv2.CalibrationMatrixValues(
            cameraMatrix, imageSize, apertureWidth, apertureHeight,
            out fovx, out fovy, out focalLength, out principalPoint, out aspectratio);

        double fovXScale = (2.0 * Mathf.Atan((float)(imageSize.Width / (2.0 * fx)))) / (Mathf.Atan2((float)cx, (float)fx) + Mathf.Atan2((float)(imageSize.Width - cx), (float)fx));
        double fovYScale = (2.0 * Mathf.Atan((float)(imageSize.Height / (2.0 * fy)))) / (Mathf.Atan2((float)cy, (float)fy) + Mathf.Atan2((float)(imageSize.Height - cy), (float)fy));

        if (widthScale < heightScale)
        {
            camera.fieldOfView = (float)(fovx * fovXScale);
        }
        else
        {
            camera.fieldOfView = (float)(fovy * fovYScale);
        }

        detector = ORB.Create(500);
        matcher = DescriptorMatcher.Create("BruteForce-Hamming(2)");

        marker = marker_obj.GetComponent<MarkerBehavior>();
        if (!marker.inititailized)
        {
            marker.Init();
        }
        marker.tracking = false;
        detector.DetectAndCompute(marker.grayMat, null, out marker.keypoints, marker.descriptors);

        previous_frameMatGray = new Mat();
        previous_points2dFrame = new List<Point2f>(4);
    }

    // Update is called once per frame
    void Update()
    {
        // Make evey marker invisible.
        marker.pattern.SetActive(false);
        for (int ii = 0; ii < marker.VirtualObjects.Length; ii++)
        {
            marker.VirtualObjects[ii].SetActive(false);
        }

        // If can't load webcame, ends.
        if (webcamTexture == null)
        {
            print("Webcam texture is empty!");
            return;
        }

        // If can't convert to opencv Mat, ends.
        Mat frameMat = OpenCvSharp.Unity.TextureToMat(webcamTexture);
        if (frameMat.Empty())
        {
            print("FrameMat is empty!");
            return;
        }
        //Debug.Log(frameMat);

        // Convert webcame frame Mat to gray.
        Mat frameMatGray = new Mat();
        Cv2.CvtColor(frameMat, frameMatGray, ColorConversionCodes.RGB2GRAY);

        // Detect and compute frame gray, if no keypoints in frame, ends.
        var frame_keypoints = new KeyPoint[] { };
        var frame_descriptors = new Mat();
        detector.DetectAndCompute(frameMatGray, null, out frame_keypoints, frame_descriptors);
        if (frame_descriptors.Empty())
        {
            RenderRawImage(frameMat);
            return;
        }

        // Mat the frame to every registered marker.
        // Filter the knnMatches with distance ratio.
        if (!marker.tracking)
        {
            OnNotTracking(frameMat, marker, frame_keypoints, frame_descriptors);
        }
        else
        {
            OnTracking(frameMat, marker);
        }

        // Render the frame Mat to unity raw image.
        RenderRawImage(frameMat);
    }

    void OnNotTracking(Mat frameRGB, MarkerBehavior marker, KeyPoint[] frame_keypoints, Mat frame_descriptors)
    {
        var matches = matcher.KnnMatch(marker.descriptors, frame_descriptors, 2);
        if (matches.Length == 0)
        {
            return;
        }
        var goodMatches = new List<DMatch>();
        var keypoint_pos_pattern = new List<Point2f>();
        var keypoint_pos_frame = new List<Point2f>();
        for (int ii = 0; ii < matches.Length; ii++)
        {
            DMatch[] knnMatches = matches[ii];
            if (knnMatches.Length < 2)
            {
                break;
            }
            DMatch bestMatch = knnMatches[0];
            DMatch betterMatch = knnMatches[1];
            float distRatio = bestMatch.Distance / betterMatch.Distance;
            if (distRatio < 0.66)
            {
                keypoint_pos_pattern.Add(marker.keypoints[bestMatch.QueryIdx].Pt);
                keypoint_pos_frame.Add(frame_keypoints[bestMatch.TrainIdx].Pt);
                goodMatches.Add(bestMatch);
            }
        }

        // If not enough good matches, this marker doesn't appear.
        if (goodMatches.Count <= 5)
        {
            print("Didn't find enough good matches!");
            return;
        }

        // Compute the homography of the marker.
        marker.homography = Cv2.FindHomography(InputArray.Create(keypoint_pos_pattern), InputArray.Create(keypoint_pos_frame), HomographyMethods.Ransac, 5);
        if (marker.homography.Empty())
        {
            return;
        }

        // Project the marker rect corners to the frame, draw the rect.
        float w = marker.grayMat.Cols;
        float h = marker.grayMat.Rows;
        var points2dPattern = new Point2f[] { new Point2f(0, 0), new Point2f(w, 0), new Point2f(w, h), new Point2f(0, h) };
        var points2dFrame = Cv2.PerspectiveTransform(points2dPattern, marker.homography);

        marker.tracking = true;
        Cv2.CvtColor(frameRGB, previous_frameMatGray, ColorConversionCodes.RGB2GRAY);
        previous_points2dFrame.Clear();
        for (int ii = 0; ii < points2dFrame.Length; ii++)
        {
            previous_points2dFrame.Add(points2dFrame[ii]);
        }
    }

    void OnTracking(Mat frameRGB, MarkerBehavior marker)
    {
        var current_frameMatGray = new Mat();
        Cv2.CvtColor(frameRGB, current_frameMatGray, ColorConversionCodes.RGB2GRAY);
        var previous_good_keypoints = Cv2.GoodFeaturesToTrack(previous_frameMatGray, 100, 0.3, 7, null, 7, false, 0.04);
        if (previous_good_keypoints.Length < 10)
        {
            marker.tracking = false;
            return;
        }

        var err = new List<float>();
        var previous_keypointMat = new Mat(previous_good_keypoints.Length, 1, MatType.CV_32FC2, previous_good_keypoints);
        var current_keypointsMat = new Mat(previous_good_keypoints.Length, 1, MatType.CV_32FC2);
        var track_status = new List<byte>();
        Cv2.CalcOpticalFlowPyrLK(previous_frameMatGray, current_frameMatGray, previous_keypointMat, current_keypointsMat, OutputArray.Create(track_status), OutputArray.Create(err), new OpenCvSharp.Size(21, 21), 3);
        int tr_num = track_status.Count;
        if (tr_num < 5)
        {
            marker.tracking = false;
            return;
        }
        var track_homography = Cv2.FindHomography(previous_keypointMat, current_keypointsMat, HomographyMethods.Ransac, 5);
        if (track_homography.Empty())
        {
            marker.tracking = false;
            return;
        }
        var current_points2dFrame = Cv2.PerspectiveTransform(previous_points2dFrame, track_homography);
        current_frameMatGray.CopyTo(previous_frameMatGray);
        previous_points2dFrame.Clear();
        for (int i = 0; i < current_points2dFrame.Length; i++)
        {
            previous_points2dFrame.Add(current_points2dFrame[i]);
        }

        if (shouldDrawRect)
        {
            for (int i = 0; i < current_points2dFrame.Length; i++)
            {
                Point2f pt1 = current_points2dFrame[i];
                Point2f pt2 = current_points2dFrame[(i + 1) % 4];
                Cv2.Line(frameRGB, pt1, pt2, Scalar.Red, 2);
            }
        }

        // Pose estimation, draw axis in frame.
        var points3dPattern = new Point3f[] { new Point3f(-0.5f, 0.5f, 0), new Point3f(0.5f, 0.5f, 0), new Point3f(0.5f, -0.5f, 0), new Point3f(-0.5f, -0.5f, 0) };
        Cv2.SolvePnPRansac(InputArray.Create(points3dPattern), InputArray.Create(previous_points2dFrame), cameraMatrix, distortionMatrix, marker.rvec, marker.tvec);

        var axisPoints = new Point3f[] { new Point3f(0, 0, 0), new Point3f(1, 0, 0), new Point3f(0, 1, 0), new Point3f(0, 0, 1) };
        var projAxisPoints = new Mat();
        Cv2.ProjectPoints(InputArray.Create(axisPoints), marker.rvec, marker.tvec, cameraMatrix, distortionMatrix, projAxisPoints, new Mat());
        if (shouldDrawAxis)
        {
            Cv2.Line(frameRGB, projAxisPoints.At<Point2f>(0, 0), projAxisPoints.At<Point2f>(0, 1), Scalar.Red, 5);
            Cv2.Line(frameRGB, projAxisPoints.At<Point2f>(0, 0), projAxisPoints.At<Point2f>(0, 2), Scalar.Green, 5);
            Cv2.Line(frameRGB, projAxisPoints.At<Point2f>(0, 0), projAxisPoints.At<Point2f>(0, 3), Scalar.Blue, 5);
        }

        for (int ii = 0; ii < marker.VirtualObjects.Length; ii++)
        {
            marker.VirtualObjects[ii].SetActive(true);
        }

        // Apply marker pose to marker.
        Mat rotMat = new Mat();
        Cv2.Rodrigues(marker.rvec, rotMat);

        var transMatrix = new Matrix4x4();
        transMatrix.SetRow(0, new Vector4((float)rotMat.At<double>(0, 0), (float)rotMat.At<double>(0, 1), (float)rotMat.At<double>(0, 2), (float)marker.tvec.At<double>(0, 0)));
        transMatrix.SetRow(1, new Vector4((float)rotMat.At<double>(1, 0), (float)rotMat.At<double>(1, 1), (float)rotMat.At<double>(1, 2), (float)marker.tvec.At<double>(1, 0)));
        transMatrix.SetRow(2, new Vector4((float)rotMat.At<double>(2, 0), (float)rotMat.At<double>(2, 1), (float)rotMat.At<double>(2, 2), (float)marker.tvec.At<double>(2, 0)));
        transMatrix.SetRow(3, new Vector4(0, 0, 0, 1));

        var invertY = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
        var invertZ = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));

        var ARMatrix = invertY * transMatrix * invertY;
        ARMatrix = ARMatrix * invertY * invertZ;

        ARMatrix = camera.transform.localToWorldMatrix * ARMatrix;
        marker.transform.localScale = new Vector3(0.25f, 0.25f, 0.25f);
        marker.transform.position = ARMatrix.GetPosition();
        marker.transform.rotation = ARMatrix.rotation;
        marker.transform.Rotate(new Vector3(-90, 0, 0));
    }

    void RenderRawImage(Mat mat)
    {
        if (rawImage.texture != null)
            Destroy(rawImage.texture);
        rawImage.texture = OpenCvSharp.Unity.MatToTexture(mat);
    }
}
