package groupxii.localproducts;

public class LocalShop {

    private String name;
    private double rating;
    private String location;

    /**
     * constructor of a local shop.
     * @param name name of the shop
     * @param rating rating of the shop
     * @param location address of the shop
     */
    public LocalShop(String name, double rating, String location) {
        this.name = name;
        this.rating = rating;
        this.location = location;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public void setRating(double rating) {
        this.rating = rating;
    }

    public String getLocation() {
        return location;
    }

    public double getRating() {
        return rating;
    }

    public String getName() {
        return name;
    }
}
