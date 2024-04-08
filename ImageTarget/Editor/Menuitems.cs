using UnityEngine;
using UnityEditor;

public class MenuItems
{
    [MenuItem("Image Target/Add AR Camera")]
    private static void AddArCamera()
    {
        Object cameraPrefab = Resources.Load("AR Camera");
        var camera = Object.Instantiate(cameraPrefab, new Vector3(0, 0, -10), Quaternion.identity);
        camera.name = "AR Camera";
    }

    [MenuItem("Image Target/Add Marker")]
    private static void AddMaker()
    {
        var markerObject = new GameObject();
        markerObject.name = "marker";
        markerObject.AddComponent<MarkerBehavior>();
    }
}