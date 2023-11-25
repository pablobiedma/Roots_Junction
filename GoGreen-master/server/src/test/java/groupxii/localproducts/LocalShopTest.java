package groupxii.localproducts;

import org.junit.Test;

import static org.junit.Assert.*;

public class LocalShopTest {

    @Test
    public void setName() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        shop.setName("The new shop");
        assertEquals("The new shop",shop.getName());
    }

    @Test
    public void setLocation() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        shop.setLocation("there");
        assertEquals("there",shop.getLocation());
    }

    @Test
    public void setRating() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        shop.setRating(5);
        assertEquals(5,shop.getRating(),0);
    }

    @Test
    public void getLocation() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        assertEquals("here",shop.getLocation());
    }

    @Test
    public void getRating() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        assertEquals(3.7,shop.getRating(),0);
    }

    @Test
    public void getName() {
        LocalShop shop = new LocalShop("The shop",3.7,"here");
        assertEquals("The shop",shop.getName());
    }
}