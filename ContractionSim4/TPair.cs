
public class TPair<T1, T2>
{

    //public T1 Item1 { get; }
    //public T2 Item2 { get; }
    //public TPair(T1 item1, T2 item2) { Item1 = item1; Item2 = item2; }

    private List<TzimtzumSimulation.Vector2> path;

    private object value;

    public TPair(List<TzimtzumSimulation.Vector2> path, object value)
    {
        this.path = path;
        this.value = value;
    }
}