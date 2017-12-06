package representation;

import gnu.trove.set.hash.TIntHashSet;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.lang3.StringUtils;
import utils.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by besnik on 7/17/17.
 */
public class CategoryHierarchy implements Serializable {
    public String label;
    public int level;
    public int node_id;

    public transient Map<String, CategoryHierarchy> parents = new HashMap<>();
    public Map<String, CategoryHierarchy> children = new HashMap<>();

    //keep the paths from all parent-child-nodes
    public TIntHashSet paths;

    //generate first the attribute representation for each category and store it
    public Map<String, Map<String, Integer>> cat_representation;

    //store the number of entities belonging to this category
    public int num_entities = 0;

    //store also the set of entities for each category
    public Set<String> entities;

    public CategoryHierarchy(String label, int level) {
        this.label = label;
        this.level = level;

        paths = new TIntHashSet();
        cat_representation = new HashMap<>();
        entities = new HashSet<>();
    }

    /**
     * Load all the children categories into a map data structure.
     * @param cats
     */
    public void loadIntoMapChildCats(Map<String, CategoryHierarchy> cats) {
        cats.put(this.label, this);
        children.values().stream().forEach(child -> child.loadIntoMapChildCats(cats));
    }

    /**
     * In the category hierarchy, assigns based on the trasnitive property all the entities that belong to a category
     * and all its super classes.
     */
    public void gatherEntities() {
        if (children != null && !children.isEmpty()) {
            for (String child_label : children.keySet()) {
                children.get(child_label).gatherEntities();
            }
        }

        if (children.isEmpty() || children == null) {
            num_entities = entities.size();
        }

        children.values().stream().forEach(cat -> entities.addAll(cat.entities));
        num_entities = entities.size();
    }

    /**
     * Find a category which is part of the children categories in the category hierarchy.
     *
     * @param label
     * @return
     */
    public CategoryHierarchy findCategory(String label) {
        if (this.label.equals(label)) {
            return this;
        }

        if (children != null && !children.isEmpty()) {
            for (String child_label : children.keySet()) {
                CategoryHierarchy result = children.get(child_label).findCategory(label);

                if (result != null) {
                    return result;
                }
            }
        }
        return null;
    }

    /**
     * Return the set of categories that belong to a certain level in the Wikipedia category taxonomy.
     *
     * @param category
     * @param level
     * @param categories
     */
    public static void getChildrenLevel(CategoryHierarchy category, int level, Set<CategoryHierarchy> categories) {
        if (category.level == level) {
            categories.add(category);
            return;
        }

        if (category.children != null && !category.children.isEmpty()) {
            for (String child_category : category.children.keySet()) {
                getChildrenLevel(category.children.get(child_category), level, categories);
            }
        }
    }

    /**
     * Construct the category graph.
     *
     * @param category_file
     * @return
     * @throws IOException
     */
    public static CategoryHierarchy readCategoryGraph(String category_file) throws IOException {
        CategoryHierarchy root = new CategoryHierarchy("root", 0);
        root.node_id = -1;

        BufferedReader reader = FileUtils.getFileReader(category_file);
        String line;

        Map<String, CategoryHierarchy> all_cats = loadAllCategories(category_file);
        while ((line = reader.readLine()) != null) {
            String[] data = line.split("\\s+");
            if (!data[1].contains("<http://www.w3.org/2004/02/skos/core#broader>") || data[2].equals(data[0])) {
                continue;
            }

            String parent_label = data[2].replace("<http://dbpedia.org/resource/Category:", "").replace(">", "");
            String child_label = data[0].replace("<http://dbpedia.org/resource/Category:", "").replace(">", "");

            CategoryHierarchy parent = all_cats.get(parent_label);
            CategoryHierarchy child = all_cats.get(child_label);

            child.level = parent.level + 1;
            parent.children.put(child_label, child);
            child.parents.put(parent_label, parent);
        }

        for (String cat_label : all_cats.keySet()) {
            CategoryHierarchy cat = all_cats.get(cat_label);
            if (!cat.parents.isEmpty()) {
                root.children.put(cat.label, cat);
                cat.parents.put(root.label, root);
            }
        }

        //we first need to break the cycles
        for (String cat_label : all_cats.keySet()) {
            CategoryHierarchy cat = all_cats.get(cat_label);

            cat.parents.keySet().removeAll(cat.children.keySet());
            if (cat.parents.size() == 1) {
                continue;
            }
            if (cat.parents.size() > 1 && cat.parents.containsKey("root")) {
                cat.parents.remove("root");
                root.children.remove(cat.label);
            } else if (cat.parents.size() == 0) {
                cat.parents.put(root.label, root);
                root.children.put(cat.label, cat);
            }
        }
        removeCyclesDFS(root);
        root.setLevels(0);
        root.ensureHierarchy();

        return root;
    }


    /**
     * Load the categories into a map datastructure.
     *
     * @param file
     * @return
     * @throws IOException
     */
    public static Map<String, CategoryHierarchy> loadAllCategories(String file) throws IOException {
        BufferedReader reader = FileUtils.getFileReader(file);
        String line;

        int node_id = 0;
        Map<String, CategoryHierarchy> all_cats = new HashMap<>();
        while ((line = reader.readLine()) != null) {
            String[] data = line.split("\\s+");
            if (data[1].contains("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")) {
                //the category doesn't, exist, add it into the root
                String cat_label = data[0].replace("<http://dbpedia.org/resource/Category:", "").replace(">", "");
                CategoryHierarchy cat = all_cats.get(cat_label);
                if (cat == null) {
                    cat = new CategoryHierarchy(cat_label, 0);
                    cat.node_id = node_id;
                    all_cats.put(cat.label, cat);
                    node_id++;
                }
                continue;
            }

            if (!data[1].contains("skos/core#broader>") || data[2].equals(data[0])) {
                continue;
            }

            String parent_label = data[2].replace("<http://dbpedia.org/resource/Category:", "").replace(">", "");
            String child_label = data[0].replace("<http://dbpedia.org/resource/Category:", "").replace(">", "");

            if (!all_cats.containsKey(parent_label)) {
                CategoryHierarchy cat = new CategoryHierarchy(parent_label, 0);
                cat.node_id = node_id;
                node_id++;
                all_cats.put(cat.label, cat);
            }

            if (!all_cats.containsKey(child_label)) {
                CategoryHierarchy cat = new CategoryHierarchy(child_label, 0);
                cat.node_id = node_id;
                node_id++;
                all_cats.put(cat.label, cat);
            }
        }
        return all_cats;
    }

    /**
     * Write the constructed category taxonomy.
     *
     * @param out_file
     * @param sb
     */
    public void printCategories(String out_file, StringBuffer sb) {
        if (sb.length() > 10000) {
            FileUtils.saveText(sb.toString(), out_file, true);
            sb.delete(0, sb.length());
        }

        String tabs = StringUtils.repeat("\t", level);
        sb.append(tabs).append(label).append("\n");

        for (String child_label : children.keySet()) {
            CategoryHierarchy cat = children.get(child_label);
            cat.printCategories(out_file, sb);
        }

        FileUtils.saveText(sb.toString(), out_file, true);
        sb.delete(0, sb.length());
    }


    /**
     * Remove parent categories whose level is higher than the minimum level.
     */
    public void ensureHierarchy() {
        if (!label.equals("root") && parents.size() > 1) {
            //if its not the root category, we check the parents of this category and remove those parents for which
            int max_level = parents.values().stream().map(x -> x.level).max((x, y) -> x.compareTo(y)).get();
            List<Map.Entry<String, CategoryHierarchy>> filtered_parents = parents.entrySet().stream().filter(x -> x.getValue().level == max_level).collect(Collectors.toList());

            parents.clear();
            filtered_parents.forEach(x -> parents.put(x.getKey(), x.getValue()));
        }

        //do this for all its children.
        for (String child_label : children.keySet()) {
            CategoryHierarchy child = children.get(child_label);
            child.ensureHierarchy();
        }
    }

    /**
     * Set the category levels such that they form a hierarchy.
     */
    public void setLevels(int level) {
        this.level = level + 1;

        //assign the level values for the children
        for (String child_label : children.keySet()) {
            CategoryHierarchy child = children.get(child_label);
            child.setLevels(this.level);
        }
    }


    /**
     * In some cases the category graph forms cycles. We break such cycles.
     */
    public static void removeCyclesDFS(CategoryHierarchy root) {
        Queue<CategoryHierarchy> cats = new LinkedList<>();
        cats.add(root);
        while (!cats.isEmpty()) {
            CategoryHierarchy cat = cats.remove();
            cat.paths.add(cat.node_id);
            if (!cat.children.isEmpty()) {
                Iterator<Map.Entry<String, CategoryHierarchy>> child_keys = cat.children.entrySet().iterator();
                while (child_keys.hasNext()) {
                    Map.Entry<String, CategoryHierarchy> cat_child = child_keys.next();
                    cat_child.getValue().paths = cat.paths;

                    if (cat.paths.contains(cat_child.getValue().node_id)) {
                        child_keys.remove();
                        continue;
                    }
                    cat_child.getValue().paths.add(cat_child.getValue().node_id);
                    cats.add(cat_child.getValue());
                }
            }
        }
    }

    public String toString() {
        return new StringBuffer().append(label).append("\t").append(level).toString();
    }


    public static void main(String[] args) throws IOException, CompressorException {
//        String base_dir = "/Users/besnik/Desktop/dbpedia/";
//        String[] args1 = {"-option", "representation", "-categories", base_dir + "skos_categories_en.nt.gz",
//                "-entity_attributes", base_dir + "entity_attributes/mappingbased_objects_en_sub.ttl", "-entity_categories", base_dir + "article_categories_sub.ttl",
//                "-out_dir", base_dir + "out", "-debug", "false"};
//        args = args1;

        String category_path = "", entity_attributes_path = "", option = "", entity_categories_path = "", out_dir = "";
        boolean debug = false;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-option")) {
                option = args[++i];
            } else if (args[i].equals("-categories")) {
                category_path = args[++i];
            } else if (args[i].equals("-entity_attributes")) {
                entity_attributes_path = args[++i];
            } else if (args[i].equals("-entity_categories")) {
                entity_categories_path = args[++i];
            } else if (args[i].equals("-out_dir")) {
                out_dir = args[++i];
            } else if (args[i].equals("-debug")) {
                debug = args[++i].equals("true");
            }
        }

        System.out.println("Read Category Graph...");
        CategoryHierarchy cat = CategoryHierarchy.readCategoryGraph(category_path);

        if (debug) {
            StringBuffer sb = new StringBuffer();
            String cat_hierarchy_file = out_dir + "/category_hierarchy.csv";
            if (FileUtils.fileExists(cat_hierarchy_file, false)) {
                new File(cat_hierarchy_file).delete();
            }
            cat.printCategories(cat_hierarchy_file, sb);
            FileUtils.saveText(sb.toString(), cat_hierarchy_file, true);
        }

        if (option.equals("representation")) {
            System.out.println("Read category to article mappings...");
            Map<String, Set<String>> entity_categories = readCategoryMappings(entity_categories_path);

            //set num entities for each category.
            Map<String, CategoryHierarchy> cats = new HashMap<>();
            cat.loadIntoMapChildCats(cats);

            entity_categories.keySet().forEach(category -> {
                CategoryHierarchy cat_node = cats.get(category);
                if (cat_node == null) {
                    return;
                }
                cat_node.entities.addAll(entity_categories.get(category));
            });

            cat.gatherEntities();

            //load the attributes for each entity
            Set<String> attribute_files = new HashSet<>();
            FileUtils.getFilesList(entity_attributes_path, attribute_files);
            Map<String, Map<String, Set<String>>> entity_attributes = loadEntityAttributes(attribute_files);

            //for each category, based on a bottom up approach, generate the attribute representation
            cat.getCategoryAttributeRepresentation(entity_categories, entity_attributes);
            cat.aggregateCategoryRepresentation();
            //save the generated category representation
            FileUtils.saveObject(cat, out_dir + "/category_hierarchy_representation.obj");

            //save also the textual representation for debugging
            cat.saveCategoryRepresentation(out_dir + "/category_hierarchy_representation.txt");
        }
    }

    /**
     * Store the category representation.
     *
     * @param out_file
     */
    public void saveCategoryRepresentation(String out_file) {
        if (children != null && !children.isEmpty()) {
            children.values().forEach(child -> child.saveCategoryRepresentation(out_file));
        }

        String cat_rep = printCategoryAttributeRepresentation();
        FileUtils.saveText(cat_rep, out_file, true);
    }

    /**
     * Load the attributes for entities in DBpedia.
     *
     * @param files
     * @return
     * @throws IOException
     */
    public static Map<String, Map<String, Set<String>>> loadEntityAttributes(Set<String> files) throws IOException {
        Map<String, Map<String, Set<String>>> entity_attributes = new HashMap<>();

        for (String file : files) {
            BufferedReader reader = FileUtils.getFileReader(file);
            String line;

            while ((line = reader.readLine()) != null) {
                String[] data = line.split("> ");
                if (data.length < 3) {
                    continue;
                }

                String entity = data[0].replace("<http://dbpedia.org/resource/", "").intern();
                String predicate = data[1].replace("<", "").intern();
                String value = data[2];
                if (value.endsWith(" .")) {
                    value = value.substring(0, value.lastIndexOf(" ")).trim();
                }

                if (!entity_attributes.containsKey(predicate)) {
                    entity_attributes.put(predicate, new HashMap<>());
                }

                if (!entity_attributes.get(predicate).containsKey(entity)) {
                    entity_attributes.get(predicate).put(entity, new HashSet<>());
                }
                entity_attributes.get(predicate).get(entity).add(value);
            }
        }

        return entity_attributes;
    }

    /**
     * Read the entity-category associations.
     *
     * @param file
     * @return
     * @throws IOException
     */
    public static Map<String, Set<String>> readCategoryMappings(String file) throws IOException {
        Map<String, Set<String>> entity_cats = new HashMap<>();
        BufferedReader reader = FileUtils.getFileReader(file);

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.replace("<http://dbpedia.org/resource/", "").replace(">", "");
            String[] parts = line.split("\\s+");
            String article = parts[0];
            String category = parts[2].replace("Category:", "");

            if (!entity_cats.containsKey(category)) {
                entity_cats.put(category, new HashSet<>());
            }
            entity_cats.get(category).add(article);
        }

        reader.close();
        return entity_cats;
    }

    /**
     * For each category, we generate a representation based on the attributes and attribute values assigned to the
     * corresponding entities belonging to the specific category. Later on, the attribute-based representation for
     * each category will be weighted according to the discriminative information provided by the attribute for the
     * category, e.g. an attribute describing the football team has higher weight than the attribute birth place for
     * a person, as it is more specific.
     * <p>
     * We will use such representation in order to find possible table candidates for alignment, in which, their
     * representations match. Furthermore, this representation will be used also to type the relation between
     * such candidate tables.
     *
     * @param entity_categories
     * @param entity_attributes
     * @return
     */
    public void getCategoryAttributeRepresentation(Map<String, Set<String>> entity_categories, Map<String, Map<String, Set<String>>> entity_attributes) {
        if (children != null && !children.isEmpty()) {
            for (String child_label : children.keySet()) {
                children.get(child_label).getCategoryAttributeRepresentation(entity_categories, entity_attributes);
            }
        }

        //get the entities of the following category
        Set<String> entities = entity_categories.get(label);
        if (entities == null) {
            return;
        }
        entities.forEach(entity -> {
            for (String attribute : entity_attributes.keySet()) {
                if (!entity_attributes.get(attribute).containsKey(entity)) {
                    continue;
                }

                Map<String, Set<String>> attribute_values = entity_attributes.get(attribute);

                if (!cat_representation.containsKey(attribute)) {
                    cat_representation.put(attribute, new HashMap<>());
                }

                for (String value : attribute_values.get(entity)) {
                    int count = cat_representation.get(attribute).containsKey(value) ? cat_representation.get(attribute).get(value) : 0;
                    count++;
                    cat_representation.get(attribute).put(value, count);
                }
            }
        });
    }

    /**
     * Aggregate the representation across the category hierarchy.
     */
    public void aggregateCategoryRepresentation() {
        if (children != null && !children.isEmpty()) {
            for (String child_label : children.keySet()) {
                CategoryHierarchy child = children.get(child_label);

                if (child.children == null || child.children.isEmpty()) {
                    mergeCategoryRepresentation(this, child);
                    continue;
                } else {
                    child.aggregateCategoryRepresentation();
                }
                mergeCategoryRepresentation(this, child);
            }
        }
    }

    /**
     * Merge two category representations.
     *
     * @param cat_parent
     * @param cat_child
     */
    private void mergeCategoryRepresentation(CategoryHierarchy cat_parent, CategoryHierarchy cat_child) {
        if (cat_parent.cat_representation.isEmpty()) {
            cat_parent.cat_representation.putAll(cat_child.cat_representation);
        } else {
            Map<String, Map<String, Integer>> child_cat_representation = cat_child.cat_representation;
            for (String key : child_cat_representation.keySet()) {
                if (!cat_representation.containsKey(key)) {
                    cat_representation.put(key, new HashMap<>());
                }

                for (String sub_key : child_cat_representation.get(key).keySet()) {
                    if (!cat_representation.get(key).containsKey(sub_key)) {
                        cat_representation.get(key).put(sub_key, 0);
                    }
                    cat_representation.get(key).put(sub_key, cat_representation.get(key).get(sub_key) + child_cat_representation.get(key).get(sub_key));
                }
            }
        }
    }

    /**
     * Print the category representation.
     *
     * @return
     */
    public String printCategoryAttributeRepresentation() {
        int total_attributes = cat_representation.size();
        if (total_attributes == 0) {
            return "";
        }
        //output the category representation
        StringBuffer sb = new StringBuffer();
        sb.append(label).append("\t").append(num_entities).append("\t").append(total_attributes);
        for (String attribute : cat_representation.keySet()) {
            int num_values = cat_representation.get(attribute).size();
            int num_assignments = cat_representation.get(attribute).values().stream().mapToInt(x -> x).sum();

            sb.append("\t").append(attribute).append("=").append(num_values).append(";").append(num_assignments);
        }
        sb.append("\n");
        return sb.toString();
    }
}
