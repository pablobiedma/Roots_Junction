package groupxii.localproducts;

import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class ReadLocalProductJsonTest {

    @Test
    public void setLocation() {
        ReadLocalProductJson json = new ReadLocalProductJson();
        json.setLocation("there");
        assertEquals("there",json.getLocation());
    }

    @Test
    public void setLocalShopList() {
        ReadLocalProductJson json = new ReadLocalProductJson();
        List<LocalShop> list = new ArrayList<>();
        LocalShop shop1 = new LocalShop("The thing",3.7,"here");
        LocalShop shop2 = new LocalShop("The shop",4.9,"behind you");
        LocalShop shop3 = new LocalShop("Shop",4.0,"there");
        list.add(shop1);
        list.add(shop2);
        list.add(shop3);
        json.setLocalShopList(list);
        assertEquals(json.getLocalShopList(),list);
    }

    @Test
    public void readLocalProductJson() throws IOException {
        ReadLocalProductJson json = new ReadLocalProductJson();
        json.readLocalProductJson();
    }

    @Test
    public void localShopToString() {
        ReadLocalProductJson json = new ReadLocalProductJson();
        List<LocalShop> list = new ArrayList<>();
        LocalShop shop = new LocalShop("Shop",4.9,"Near");
        list.add(shop);
        json.setLocalShopList(list);
        assertEquals(json.localShopToString()," SHOP:  " + shop.getName() + "  - LOCATED AT:  " + shop.getLocation() + "  - RATING:  " + shop.getRating() + ", ");
    }

    @Test
    public void clearList() {
        ReadLocalProductJson json = new ReadLocalProductJson();
        List<LocalShop> list = new ArrayList<>();
        List<LocalShop> clearlist = new ArrayList<>();
        LocalShop shop1 = new LocalShop("The thing",3.7,"here");
        LocalShop shop2 = new LocalShop("The shop",4.9,"behind you");
        LocalShop shop3 = new LocalShop("Shop",4.0,"there");
        list.add(shop1);
        list.add(shop2);
        list.add(shop3);
        json.setLocalShopList(list);
        json.clearList();
        assertEquals(clearlist,json.getLocalShopList());
    }
}