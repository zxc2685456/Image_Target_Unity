using UnityEngine;
using OpenCvSharp;
using UnityEngine.UI;
using UnityEngine.Assertions;

public class MarkerBehavior : MonoBehaviour
{
    public GameObject pattern; // This should be a sprite.
    public GameObject[] VirtualObjects;

    [HideInInspector] public bool inititailized = false;
    [HideInInspector] public Mat grayMat;
    [HideInInspector] public KeyPoint[] keypoints;
    [HideInInspector] public Mat descriptors;
    [HideInInspector] public Mat homography;
    [HideInInspector] public bool tracking;
    [HideInInspector] public Mat rvec;
    [HideInInspector] public Mat tvec;

    public void Init()
    {
        grayMat = new Mat();
        keypoints = new KeyPoint[] { };
        descriptors = new Mat();
        rvec = new Mat();
        tvec = new Mat();

        var sprite = pattern.GetComponent<SpriteRenderer>().sprite;
        var texture = new Texture2D((int)sprite.rect.width, (int)sprite.rect.height);
        var pixels = sprite.texture.GetPixels(0, 0, (int)sprite.rect.width, (int)sprite.rect.height);
        texture.SetPixels(pixels);
        texture.Apply();

        grayMat = OpenCvSharp.Unity.TextureToMat(texture);
        Assert.AreNotEqual(grayMat.Cols, 0);
        Assert.AreNotEqual(grayMat.Rows, 0);
        Cv2.CvtColor(grayMat, grayMat, ColorConversionCodes.RGB2GRAY);

        pattern.SetActive(false);
        for (int i = 0; i < VirtualObjects.Length; i++)
            VirtualObjects[i].SetActive(false);

        inititailized = true;
    }
}
